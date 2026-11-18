# QSA indexer fused select: implementation plan

Living plan for [SILOTIGER-1481](SILOTIGER-1481-description.md): stop writing the `[M, columns]` fp32 score buffer and the separate top-k in the QSA indexer for prefill and mixed steps. Check a box when a step is done. Put measured numbers in [Results](#results) and dated notes in [Log](#log). Leave the prose in place when a step finishes.

The mechanism comes from the design spike in [design-spike.md](design-spike.md). That spike picked option 1, a fused split-local top-512 followed by a split-aware merge. This plan implements that choice, gates it against the ticket's acceptance criteria, and records when the fallback applies.

## Locked decisions

Change these only by editing this section and adding a log line. Code, tests, and the write-up follow this section as it stands.

### Test environment

- Interpreter: `/workspaces/aiter/flydsl_venv/bin/python` (Python 3.14.4, torch 2.12.0+rocm7.14.0). aiter is an editable install in that venv.
- **All code that runs on a GPU (kernels, op tests, probes, benchmarks, and serving runs) must run in `flydsl_venv` and be queued through `gpuq` as specified in [`gpuq.md`](gpuq.md).** Set `GPUQ_USER` to the job owner and launch with `gpuq --num-gpus=1` (or `--num-gpus=N` when a step needs more, for example TP2 serving). Do not start GPU work with a bare interpreter. Do not pin a device with `HIP_VISIBLE_DEVICES`; `gpuq` assigns it. Do not install or restart the gpuq daemon. A job that skips `gpuq` can be SIGTERMed while eviction is on.
- Kernels must be compiled before the related work is queued with `gpuq`. A test, bench, probe, or serving run is queued only after the kernels it launches are already compiled, so the queued job does not spend its GPU lease on a first-time compile.
- Launch prefix:

```bash
export GPUQ_USER="<owner>"
gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python <script> [args]
```

- Before each timed session, confirm the assigned GPU is idle. `rocm-smi --showuse` must report 0%, `rocm-smi --showpids` must show no process on that GPU, and `gpuq --status` must show no other lease. gpuq's GPU index is not rocm-smi's index. Match them by PCI bus, read inside the job with `hipDeviceGetPCIBusId` on device 0. Discard and retry a session that lands on a busy GPU. One process holds the device for the whole session.
- Record for every timed session: date, `get_gfx()` and `get_gfx_runtime()`, gpuq index, rocm-smi index, PCI bus, device name and id, and the idle check.
- Branch: `samaario/qsa-indexer-fused-select`. Correctness tests stay importable and runnable under `pytest` through the same `gpuq` prefix.

### Hardware

- MI355X (gfx950) is the target and the acceptance device. Every gate in [Acceptance gates](#acceptance-gates) is measured there.
- MI300X (gfx942) is measured. The fused path is enabled on gfx942 only for the shapes where it beats the existing path there. Otherwise gfx942 keeps the current `qsa_indexer_block_ids` path.
- LDS budget: gfx950 has 160 KiB per CU and gfx942 has 64 KiB.

### Mechanism (from the design spike)

- Option 1 is the design: fuse scoring with a split-local top-512, then run a split-aware merge. Each workgroup owns a row tile and a column span. It keeps each row's local top-512 over that span, fills the merge's first-level histogram while scoring, and writes candidates instead of fp32 scores. A merge then selects each row's final 512 from the splits.
- Option 2 (score twice with a histogram, never store scores) is out. On the gfx950 spike sessions, two plain score passes exceeded today's whole step by 15.5% at 128K and 21.7% at 256K, before any histogram cost. Do not reopen it without new evidence recorded in [Log](#log).
- The fallback is to keep one score pass and store order-preserving 16-bit keys instead of fp32, with an exact fp32 rescore of the boundary bucket. It is used only if [Phase 6](#6-performance-gate-against-the-arms) shows option 1 missing the kernel gate after tuning. Decide that in [Decision record](#decision-record).
- Spike evidence carries over with limits. The spike measured the in-tree 64×64 logits scorer (`qsa_flydsl_layout.py`, compressed page 16) with dummy LDS, not this indexer's 16-row tile. It bounded the merge by today's top-k. Its relevant gfx950 numbers are:
  - A 32 KiB candidate array slowed scoring by 147.3 µs at 128K and 357.4 µs at 256K, against top-k-plus-expand budgets of 992.0 µs and 1781.3 µs.
  - A 64 KiB array launched but slowed 128K scoring from 1426.1 µs to 2668.4 µs. That 1242.3 µs loss exceeds the 992.0 µs budget.
- Phase 1 re-measures the real tile before kernel work starts.

### Candidate state

- Default per-workgroup candidate LDS is at most 32 KiB. A 16-row tile with 512 eight-byte candidates per row is 64 KiB. The spike showed that size costs more than the fusion saves on the 64×64 scorer. The kernel therefore must use one of these layouts:
  - Keep 8 rows of candidates in LDS at a time.
  - Store 4-byte candidates (a 16-bit order-preserving key and a 16-bit column offset within the split). Split width is at most 65536 columns, so that offset fits.
  - Hold part of the state in registers.
- Pick one in Phase 1 from measured occupancy on the real tile. Raise the 32 KiB cap only if Phase 1 shows that the 64 KiB cost stays under that case's top-k-plus-expand budget, and log the change.
- Global candidate workspace per row is `splits × 512 × candidate bytes`. That is 16 KiB at 128K with four 8-byte splits, instead of 128 KiB of fp32 scores.

Chosen 2026-11-18 from the strided-cache probe. Each candidate is a 16-bit order-preserving key and a 16-bit column offset, and all 16 rows of the tile keep a local top-512 in LDS. That is 32 KiB. The 16-row tile's own LDS is 34400 bytes (Q 16624, K 17392, plus the visible, request and live arrays), so the workgroup total is 67168 bytes. 64 KiB of candidates, which is 16 rows of 8-byte state, is over every gfx950 budget, so the cap stays 32 KiB. The 32 KiB dead-array slowdown stays under those budgets, with 317.8 µs to spare at `p256k` (1590.8 µs against 1908.6 µs). A live top-512 update that spends that margin is a Phase 6 retune. The 16-bit key has to keep two scores distinct when they differ by more than the fp32 tie bound.

The split count follows the trimmed table, `cdiv(max_seq_len, 1568)` pages of 392 columns, and each split stays within 65536 columns:

| `max_seq_len` | Pages | Columns | Splits | Columns per split |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 6 | 2352 | 1 | 2352 |
| 32768 | 21 | 8232 | 1 | 8232 |
| 131072 | 84 | 32928 | 1 | 32928 |
| 262144 | 168 | 65856 | 2 | 32928 |

Workspace per row is `splits × 512 × 4` bytes: 2048 through 128K and 4096 at 256K. gfx942 has 31136 bytes free beside the tile, so this 32 KiB state does not launch there. The fused path stays on the current indexer on gfx942 until Phase 6 measures a state that fits in 16 KiB.

### Base code and contract

- Start from the in-tree #5996 indexer: `qsa_indexer_block_ids` and `qsa_indexer_score_and_select` in `aiter/ops/flydsl/qsa_paged.py`, with kernels in `aiter/ops/flydsl/kernels/qsa/indexer_select.py`. Keep its multi-row tiling: 16 query rows per workgroup, `_ROWS_BLOCK_N = 64` columns, up to four column tiles per workgroup, and row tiles that may span requests. Key reuse across rows is what makes this scorer fast, and the fused kernel must keep it.
- The new entry point is the ticket's `qsa_indexer_select`, with names and argument split as proposed there.
  - Inputs:
    - `q`: bf16 `[M, 4, 128]`
    - `k_cache`: bf16 `[num_pages, page, 1, 128]`
    - `page_table`, `token_to_req`, `query_positions`, `context_lens`
    - Keyword-only host ints `num_decode_rows` and `max_seq_len`
  - Outputs:
    - Optional `block_ids`: int32 `[M, 512]`, any order, `-1` padded
    - `token_ids`: int32 `[M, 2051]`, the expanded ids and tail, `-1` padded
    - Optional caller-owned `workspace`, sized once from `max_num_batched_tokens` and `max_model_len`, and reusable across calls without a memset
- Expand is folded into the fused path. Selected blocks are written as token ids directly.
- When `n_i <= 512`, take all visible blocks without scoring them. Reuse the existing emit path.
- Missing pages keep today's behavior: a block on a missing page is dropped, and kept ids stay packed in front of `-1`.
- The host picks the variant per step from `max_seq_len`, row counts, and `num_decode_rows`. These steps run outside CUDA graphs, so a host choice is allowed.
- Existing entry points (`qsa_indexer_block_ids`, `qsa_select_paged_tokens`) keep their numerics and launches until Phase 7 switches dispatch. Then the old path stays as the fallback for shapes or archs the fused path does not win.
- The merge workspace zeroes itself, as aiter#5525's does. No caller memset and no graph-replay memset.

### Scope

- Primary: prefill and mixed prefill/decode steps at 8K, 32K, 128K and 256K, plus 1M where `max_model_len` allows. Steps have up to 16384 rows, single and multiple prompts, and a mixed step puts its decode rows first.
- Secondary, measure only: pure decode, 1 to 64 rows and MTP rows, at 8K to 256K under graph replay. No gate, and no decode-specific kernel in this ticket.
- Out of scope:
  - Specialized decode kernels (vllm#65188 `--monokernel`)
  - FP8 and MXFP4 keys (`compressed_k_scale`)
  - `stacked_decode`
  - The 64×64 logits scorer path in `aiter/ops/qsa.py`, except as a comparison arm

### Geometry

Performance and correctness use real vLLM geometry unless a step says otherwise:

- Compressed page 392, from a 1568-token main page divided by 4.
- Page tables padded to `max_model_len` 262144, which is 168 pages or 65,856 columns.
- A strided per-layer cache view whose span exceeds 4 GiB.
- vLLM's row order and chunking: decode rows first, and `max_num_batched_tokens` 16384.
- A timed case is one scheduler step of at most 16384 rows. A longer prompt is its last step, with positions from `context - rows` through `context - 1`. Earlier steps of that prompt are not summed into the case. A 1M prompt is out of Phase 0 because `max_model_len` is 262144.

The spike's identity page table, compressed page 16 and contiguous K are not acceptance geometry.

### Correctness reference

- The oracle is `qsa_oracle` in `op_tests/test_flydsl_qsa.py`: fp32 block-causal ReLU-sum scores, exact top-512, expand and tail. Validate it once against an fp64 rescore.
- The fp32 tie bound is 1.7e-6. On the seeded 8K, 32K, and padded-prefix rescores the largest `|fp32 - fp64|` was 1.62e-6, and the top-512 sets matched. A kernel block that is not in the oracle set is accepted only when its oracle score is within this bound of the 512th score. The oracle-versus-fp64 check uses twice that bound.
- Per-row gate: set equality with the oracle, except for blocks whose scores tie within the fp32 error bound at the 512th score. This follows vllm#65188 commit `3c5e29ffcd`: at least 99.9% block agreement with fp64, and every differing block within twice the bound of the 512th score.
- Never use another production kernel, including today's `qsa_indexer_block_ids`, as the oracle. Equality with the old path is a regression signal, not the gate.

### Timing protocol

- Use CUDA events around each stage, summed across a step's chunks or launches. Report `T_score`, `T_topk` or `T_merge`, `T_expand`, and `T_step` per QSA layer per step, plus chunk count and bytes written and read.
- Use 5 warmup steps, then 10 timed steps, and report the median. Interleave arms within a session, and repeat each arm at least twice per session.
- Time warm and cold caches. For a cold run, flush L2 by touching a buffer larger than the last-level cache between steps.
- Time sparse GQA alongside the indexer so its share is known.

### Acceptance gates

These are the ticket's gates, restated so progress can be checked against them:

1. Kernel: indexer time per QSA layer per step, covering scoring, selection and expand including any chunking, must beat the faster of the vllm#65188 path and aiter#5996. That applies to prefill and mixed steps at 32K, 128K and 256K, single and multi-prompt, with and without decode rows. At 8K and below, it must be within noise of the faster arm.
2. Correctness: the set gate above on every measured case, including padded tables, missing pages, wide cache views and mixed steps.
3. Serving: Qwen3.8-Flash-Next TP2 on MI355X through the vLLM opt-in path. Median TTFT at 32K and 128K inputs (c1, c4, c16) must be lower than at the vllm#65188 head with its prefill scorer. Median ITL must not get worse, and GSM8K and GPQA Diamond must be unchanged.
4. Decode: a table with no gate.
5. Early close: closing with a profile only is valid if Phase 0 shows the score buffer and top-k are a small share of long-context prefill indexer time. That means top-k-plus-expand under 15% of `T_step` at 128K and 256K, on the real indexer path. Read the warm-2 medians of single-prompt `p128k` and `p256k` on Arm A. It fires only when both are under 0.15. Mixed and multi-prompt shares, and cold-cache repeats, are recorded and do not fire it. The spike's gfx950 shares were 0.404 and 0.375 on the 64×64 scorer. Applied 2026-11-18: the shares are 0.454 and 0.334, so the profile-only close is not taken.

### Coordination

- aiter#5525, #5677 and #5996 all edit `aiter/ops/flydsl/topk/topk_per_row.py`. Agree a merge order with their owners before changing that file. Johannes Graner now carries #5525.
- aiter#5525's `split_topk_merge.py` and `fp8_paged_mqa_local_topk.py` are not in this tree. Port or reimplement the split-aware merge here. Do not depend on an unmerged branch.
- Build on the #5996 and #6083 heads already in this tree.

### vLLM checkouts

Both competitor arms are open pull requests on `vllm-project/vllm`. The local checkout at `/workspaces/vllm` (`6ad2e06c4a`) contains neither. Fetch the named commit into `flydsl_venv` before timing that arm. Do not time a different kernel in its place.

- [vllm#65188](https://github.com/vllm-project/vllm/pull/65188), branch `rocm/qwen38-fn-wide-mono`. Phase 0 arm B is commit `3c5e29ffcd` ("Qwen3.8 QSA: MFMA indexer scorer for prefill chunks"), not the PR head. The head (`0c1f099a42` on 2026-11-12) is specialized decode (`--monokernel`) and its latest commit reverts the stock QSA scoring and prefill changes, so a head checkout would not measure the prefill scorer this baseline is comparing. The serving comparison in Phase 8 stays against the PR head, which is a different checkout from this kernel pin.
- [vllm#62589](https://github.com/vllm-project/vllm/pull/62589), branch `qsa-amd-indexer-tiled`. Phase 0 arm C is the tile scorer on this branch. The tile-scorer commit is `e7a2acb5e3`. The PR head on 2026-11-12 is `348ca71d6c`, which adds a gfx942 single-row decode score and a test on top of that commit. Arm C runs `e7a2acb5e3` when the later commits are not required to build the tile scorer.

## Phases

### 0. Matched baseline on the real indexer

- [x] Add `op_tests/bench_qsa_indexer_select.py`, importable with a `__main__` guard. It builds the ticket's cases in real geometry (see [Geometry](#geometry)) and times each arm per stage (see [Timing protocol](#timing-protocol)). Written, parses, and passes ruff and black. It has not run on a GPU yet.
- [x] Arm A: aiter#5996 `qsa_indexer_block_ids` plus `qsa_expand_block_indices`, as in this tree. Measured 2026-11-12. The Baseline table holds the warm-2 medians. Each case is the bench's one chunk, not a summed prompt.
- [x] Arm B: the vLLM AMD MFMA prefill scorer at vllm#65188 commit `3c5e29ffcd` on branch `rocm/qwen38-fn-wide-mono` (see [vLLM checkouts](#vllm-checkouts)). It needs that commit in `flydsl_venv`. If it is not available, record why and leave the cell empty. Do not substitute another kernel. Measured in the interleaved session. At `p256k` the MFMA scorer does not run.
- [x] Arm C: vllm#62589's Triton tile scorer, commit `e7a2acb5e3` on branch `qsa-amd-indexer-tiled`, only if it is still open and runnable. The PR was open on 2026-11-12. Measured in the interleaved session.
- [x] Cases. The bench's ten cases are the Phase 0 set, and each one is one scheduler step (see [Geometry](#geometry)). A prompt longer than 16384 tokens is its last step, not the sum of every step of that prompt. 1M is not in the set. Two- and four-prompt steps and the mixed steps are at 128K. The same shapes at 32K and 256K stay in the Phase 6 matrix.
  - One prompt of 8K, 32K, 128K and 256K, plus 1M where allowed, as 16384-token chunks
  - Two and four prompts per step
  - A prompt chunk with 3, 16 and 64 decode rows
  - A prompt chunk with MTP decode rows, 4 per request
- [x] Record the per-stage split and the top-k-plus-expand share. Apply gate 5. If it fires, record the profile-only close in [Decision record](#decision-record) and stop. Applied on the interleaved warm-2 medians. Arm A shares are 0.454 at `p128k` and 0.334 at `p256k`. Gate 5 does not fire, and Phase 1 starts.
- [x] Fill the [Baseline](#baseline) table and log the session. The table is the interleaved warm-2 session. Each row is one scheduler step.

### 1. Candidate-state probe on the real tile

- [x] Make a spike copy of the 16-row prefill scorer, `build_qsa_indexer_prefill_scores_module`, with a compile-time candidate array. Do not change the production module.
- [x] Faithfulness: the zero-byte copy must match the unmodified scores on 32K, with median `T_score` within 5% of the same-session unmodified median.
- [x] Time candidate arrays of 16 KiB, 32 KiB and 64 KiB at 128K and 256K on gfx950. Compare each slowdown with that case's Phase 0 top-k-plus-expand budget. The `p256k` budget is 1955.5 µs, measured on the 168-page table, which is the trimmed width at `max_model_len`. The Baseline `p128k` budget of 1724.4 µs is the full 65856-column pad, while a prefill step trims to `cdiv(max_seq_len, 1568)` pages (32928 columns at 128K). Keep the 32 KiB candidate cap unless an Arm A timing of `p128k` on that prefix still covers the 64 KiB slowdown. Take that timing in this session.
- [x] Repeat on gfx942 and record any launch failure as a fit result. Measured 2026-11-18, see [Candidate-state probe, gfx942](#candidate-state-probe-gfx942). 16 KiB launches. 32 KiB and 64 KiB do not.
- [x] Choose the candidate layout under [Candidate state](#candidate-state): 8-byte or 4-byte candidates, rows held in LDS, and split count per context bucket. Log the choice. Chosen 2026-11-18: 4-byte candidates for all 16 rows (32 KiB), one split through 128K and two at 256K.

### 2. Oracle and correctness tests

- [x] Validate `qsa_oracle` against an fp64 rescore on the 8K and 32K cases and on one padded-table case. Record the fp32 error bound used for ties. Measured 2026-11-18. The bound is 1.7e-6, and all three top-512 sets matched.
- [x] Add a `qsa_indexer_select` set-gate test to `op_tests/test_flydsl_qsa.py`. Cover padded tables to `max_model_len`, missing pages, a cache view spanning more than 4 GiB, row tiles spanning requests, mixed steps with decode rows first, MTP rows, and `n_i <= 512`. Added 2026-11-18. Every case matched the oracle on gfx950.
- [x] Add a test that the workspace is reused across calls without a memset, and that results do not depend on its previous contents. The same token ids come back from a NaN-filled buffer and from a buffer filled with 12345.
- [x] Tests skip cleanly on archs where the fused path is gated off. `qsa_indexer_select_supported` rejects gfx942 and any arch outside gfx950, and the set-gate returns before launching.

### 3. Fused score and split-local top-512 kernel

- [x] Add a new module in `aiter/ops/flydsl/kernels/qsa/` for fused score and local top-512. It keeps the 16-row multi-row tile, the bf16 MFMA scoring, and the 4-head ReLU sum.
- [x] Assign each workgroup one row tile and one column span. Pick the split count per `max_seq_len` bucket so the GPU fills with row tiles, using few splits that are each much wider than 512.
- [x] Maintain each row's local top-512 over the span in the Phase 1 layout. Write candidates (key and column) to the global workspace. Do not write fp32 scores.
- [x] Fill the merge's first-level radix histogram while scoring.
- [x] Mask invisible and incomplete blocks, and drop blocks on missing pages, matching current semantics.
- [x] Correctness: the Phase 2 tests pass with a reference merge in torch, before the real merge exists.

### 4. Split-aware merge with folded expand

- [x] Write a merge kernel that reads each row's split candidates and the first-level histogram, starts at the second radix pass, and selects the final 512. Exact ties at the 512th score may resolve to either block.
- [x] Fold expand into the merge: write `token_ids [M, 2051]` directly, with each block's 4 tokens and the incomplete tail, `-1` padded. Optionally also write `block_ids [M, 512]`.
- [x] Make the workspace self-zeroing, so the merge clears the histogram it consumed for the next call.
- [x] Phase 2 set-gate tests pass end to end.

### 5. Host entry point and dispatch

- [ ] Add `qsa_indexer_select` in `aiter/ops/flydsl/qsa_paged.py`, following the [contract](#base-code-and-contract). Validate shapes, strides, dtypes, the `heads == 4` contract, arch, and workspace size, and reject anything unsupported with a clear reason.
- [ ] Per-step variant choice on the host:
  - Rows no wider than 512 take the emit path.
  - Long rows take fused plus merge.
  - Shapes the fused path does not win take the existing path.
  - Write the thresholds down as measured, not guessed.
- [ ] Mixed steps: measure decode rows in the prefill launch against a separate small launch. Answer the ticket's open question in [Log](#log) and pick the faster option.
- [ ] Add a helper that sizes the workspace once from `max_num_batched_tokens` and `max_model_len`.

### 6. Performance gate against the arms

- [ ] Rerun the Phase 0 bench with the fused arm added, on an idle gfx950, in one session with interleaved repeats, warm and cold.
- [ ] Apply gate 1 per case and fill the [Kernel gate](#kernel-gate) table.
- [ ] Tune the split count, tile count and candidate layout on the losing cases. Record each tuning change and its effect in [Log](#log).
- [ ] Measure gfx942. Enable the fused path there only for the cases where it wins, and record that set.
- [ ] If option 1 still misses gate 1 on 128K or 256K after tuning, record that in [Decision record](#decision-record) and plan the 16-bit fallback as a follow-up subtask list here.

### 7. Integration in aiter

- [ ] Route `qsa_select_paged_tokens` and `qsa_indexer_block_ids` callers to the fused path for supported shapes, behind the Phase 5 dispatch. Keep the old path for every other shape and arch.
- [ ] Run the existing `op_tests/test_flydsl_qsa.py` suite and the new tests in full through `gpuq`.
- [ ] Before editing `topk_per_row.py`, agree the merge order with the #5525 and #5677 owners. If the merge kernel does not need that file, record that.

### 8. Serving validation

- [ ] Wire the fused entry into the vLLM opt-in path (`--attention-config.qwen4_exp_qsa_backend flydsl`, vllm#59437) on a vLLM checkout in `flydsl_venv`.
- [ ] Run Qwen3.8-Flash-Next TP2 on MI355X through `gpuq --num-gpus=2`, comparing median TTFT at 32K and 128K (c1, c4, c16) against the vllm#65188 head with its prefill scorer. Also record median ITL.
- [ ] Recheck the earlier mixed-step TTFT regressions (128K c16, 60K c16, 8K c32).
- [ ] Confirm GSM8K and GPQA Diamond are unchanged. Fill the [Serving gate](#serving-gate) table.

### 9. Decode table

- [ ] Time 1 to 64 rows and MTP rows at 8K to 256K under graph replay with padded tables. Record scoring, selection and merge time for the fused design against the current paths. Fill the [Decode table](#decode-table).

### 10. Close

- [ ] Write the outcome, the gates met, and the margins into [Decision record](#decision-record).
- [ ] Log the date. Leave spike copies and probes out of production dispatch.

## Results

### Environment confirmation

| Session | Date | `get_gfx()` | gpuq / rocm-smi / bus | Device | Idle |
| --- | --- | --- | --- | --- | --- |
| Arm A | 2026-11-12 | gfx950 / gfx950 | 6 / 6 / 0000:e5:00.0 | MI355X `0x75a3` | use 0%, no other pids. The first grant, bus 0000:15:00.0, had other pids and was discarded (exit 86). K pool was uninitialized. |
| A/B/C | 2026-11-12 | gfx950 / gfx950 | 4 / 7 / 0000:f5:00.0 | MI355X `0x75a3` | use 0%, no other pids. The first grant, bus 0000:15:00.0, had other pids and was discarded (exit 86). Live pages filled. |
| Candidate probe, gfx942 | 2026-11-18 | gfx942 / gfx942 | no gpuq (not available on this node) / 7 / 0000:cf:00.0 | MI300X `0x74a1`, 304 CUs | use 0%, no other pids. `HIP_VISIBLE_DEVICES=7`. The probe's own idle check passed. |
| Candidate probe, strided cache | 2026-11-18 | gfx950 / gfx950 | 0 / 1 / 0000:15:00.0 | MI355X `0x75a3` | use 0%, no other pids. Grants on 0000:95:00.0 (use 2%) and 0000:05:00.0 (other pids) exited 86 and were discarded. |

### Baseline

Median microseconds, warm-2, from one interleaved session on gfx950. Each case is one scheduler step. A prompt longer than 16384 tokens is its last step, not the sum of that prompt's steps. The 5460 live K pages hold random bf16. An earlier Arm A session left that pool uninitialized, and its top-k times were several times higher; those numbers are not in this table. Arm B at `p256k` uses the per-row scorer, not MFMA. Score bytes are the full-width fp32 buffer. Gate 5 reads the Arm A `p128k` and `p256k` shares in this table and does not fire.

| Case | Arm | Chunks | Score bytes | `T_score` | `T_topk` | `T_expand` | `T_step` | top-k+expand share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p8k | aiter#5996 | 17 | 2058.0 MiB | 328.0 | 438.4 | 83.5 | 849.9 | 0.614 |
| p8k | vllm#65188 | 1 | 2058.0 MiB | 112.1 | 57.4 | 37.8 | 207.3 | 0.459 |
| p8k | vllm#62589 | 17 | 2058.0 MiB | 784.6 | 322.0 | 125.7 | 1232.3 | 0.363 |
| p32k | aiter#5996 | 33 | 4116.0 MiB | 954.3 | 820.6 | 146.9 | 1921.8 | 0.503 |
| p32k | vllm#65188 | 4 | 4116.0 MiB | 584.2 | 240.3 | 85.5 | 909.9 | 0.358 |
| p32k | vllm#62589 | 33 | 4116.0 MiB | 1603.6 | 845.8 | 223.6 | 2673.0 | 0.400 |
| p128k | aiter#5996 | 33 | 4116.0 MiB | 2072.9 | 1600.1 | 124.3 | 3797.3 | 0.454 |
| p128k | vllm#65188 | 16 | 4116.0 MiB | 2694.2 | 1342.0 | 141.2 | 4177.5 | 0.355 |
| p128k | vllm#62589 | 33 | 4116.0 MiB | 2814.0 | 1444.3 | 219.0 | 4477.4 | 0.372 |
| p256k | aiter#5996 | 33 | 4116.0 MiB | 3905.9 | 1833.2 | 122.3 | 5861.4 | 0.334 |
| p256k | vllm#65188 | 32 | 4116.0 MiB | 63992.6 | 2186.8 | 212.8 | 66392.3 | 0.036 |
| p256k | vllm#62589 | 33 | 4116.0 MiB | 4500.9 | 2281.9 | 222.8 | 7005.6 | 0.358 |
| two128k | aiter#5996 | 33 | 4116.0 MiB | 2138.3 | 1643.4 | 126.0 | 3907.6 | 0.453 |
| two128k | vllm#65188 | 16 | 4116.0 MiB | 2785.6 | 1368.8 | 145.8 | 4300.2 | 0.352 |
| two128k | vllm#62589 | 33 | 4116.0 MiB | 3093.3 | 1469.4 | 228.1 | 4790.8 | 0.354 |
| four128k | aiter#5996 | 33 | 4116.0 MiB | 2239.6 | 1674.0 | 125.7 | 4039.3 | 0.446 |
| four128k | vllm#65188 | 16 | 4116.0 MiB | 2823.8 | 1385.0 | 144.4 | 4353.3 | 0.351 |
| four128k | vllm#62589 | 33 | 4116.0 MiB | 3186.6 | 1475.8 | 227.5 | 4889.9 | 0.348 |
| mix3 | aiter#5996 | 33 | 4116.0 MiB | 2209.7 | 1606.7 | 125.6 | 3942.1 | 0.439 |
| mix3 | vllm#65188 | 17 | 4116.0 MiB | 2752.8 | 1386.2 | 176.3 | 4315.3 | 0.362 |
| mix3 | vllm#62589 | 33 | 4116.0 MiB | 3187.9 | 1444.6 | 226.4 | 4859.0 | 0.344 |
| mix16 | aiter#5996 | 33 | 4116.0 MiB | 2758.8 | 1615.5 | 125.1 | 4499.4 | 0.387 |
| mix16 | vllm#65188 | 17 | 4116.0 MiB | 2783.5 | 1359.3 | 174.2 | 4317.0 | 0.355 |
| mix16 | vllm#62589 | 33 | 4116.0 MiB | 3828.8 | 1435.8 | 225.2 | 5489.8 | 0.303 |
| mix64 | aiter#5996 | 33 | 4116.0 MiB | 3231.5 | 1612.2 | 124.9 | 4968.6 | 0.350 |
| mix64 | vllm#65188 | 17 | 4116.0 MiB | 2838.9 | 1364.7 | 152.5 | 4356.2 | 0.348 |
| mix64 | vllm#62589 | 33 | 4116.0 MiB | 3835.5 | 1438.3 | 226.0 | 5499.8 | 0.303 |
| mtp | aiter#5996 | 33 | 4116.0 MiB | 2135.2 | 1608.4 | 126.0 | 3869.6 | 0.448 |
| mtp | vllm#65188 | 17 | 4116.0 MiB | 2752.8 | 1386.7 | 178.6 | 4318.1 | 0.362 |
| mtp | vllm#62589 | 33 | 4116.0 MiB | 3088.1 | 1430.1 | 227.7 | 4745.8 | 0.349 |

### Candidate-state probe

Median `T_score` in microseconds, warm-2, on a strided per-layer cache (page stride 262176, span 4295067584 bytes). "fail" means the kernel did not launch. On `p128k` and `p256k` the 0 B column is the unmodified scorer from the same session. The 0-byte copy is the `p32k` row. Each budget is this session's Arm A top-k plus expand on that same cache.

| Arch | Case | 0 B | 16 KiB | 32 KiB | 64 KiB | top-k+expand budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gfx950 | p32k 0 B copy | 711.4 | | | | |
| gfx950 | p32k unmodified | 704.8 | | | | |
| gfx950 | p128k pad | 2027.6 | 2338.7 | 2980.5 | 5242.2 | 1721.2 |
| gfx950 | p128k trim | 1537.0 | 1779.2 | 2251.8 | 3888.1 | 1499.5 |
| gfx950 | p256k | 3849.8 | 4279.5 | 5440.6 | 9689.3 | 1908.6 |

Layout chosen from this table: 4-byte candidates for all 16 rows, 32 KiB per workgroup. See [Candidate state](#candidate-state).

### Fused select, Phase 3

Set-gate on gfx950, 2026-11-18. `select err` is the fraction of rows outside the 1.7e-6 window. The `select us` column is this run's `run_perftest` median on the gate's small batches. The top-512 update is a serial scan of the 512-slot list, so these times are not a serving measurement.

| case | rows | columns | select us | select err |
| --- | ---: | ---: | ---: | ---: |
| pad | 32 | 65856 | 29593 | 0 |
| missing | 16 | 2352 | 21163 | 0 |
| wide | 16 | 2352 | 27944 | 0 |
| span | 16 | 2352 | 54653 | 0 |
| mixed | 16 | 2352 | 128912 | 0 |
| mtp | 8 | 2352 | 58133 | 0 |
| emit | 16 | 392 | 6.75 | 0 |
| workspace | 32 | 65856 | 29009 | 0 |

### Fused select, Phase 4

Set-gate on gfx950, 2026-11-18, after the merge replaced the torch reduction and the Triton expand on the long-row path. `select err` is the fraction of rows outside the 1.7e-6 window. Every case is one span, so the merge keeps the scorer's top-512. The `select us` column is still the serial 512-slot scan on these small batches, not a serving measurement. Emit stays on the one-page path.

| case | rows | columns | select us | select err |
| --- | ---: | ---: | ---: | ---: |
| pad | 32 | 65856 | 29372 | 0 |
| missing | 16 | 2352 | 21125 | 0 |
| wide | 16 | 2352 | 28080 | 0 |
| span | 16 | 2352 | 54261 | 0 |
| mixed | 16 | 2352 | 134729 | 0 |
| mtp | 8 | 2352 | 53216 | 0 |
| emit | 16 | 392 | 5.52 | 0 |
| workspace | 32 | 65856 | 29004 | 0 |

### Candidate-state probe, gfx942

Median `T_score` in microseconds, two repeats (`r1 / r2`), each the median of 10 after 5 warmups. "fail" means the launch raised `hipErrorIllegalState`, checked in a fresh process. The gfx942 scorer uses the same probe copy as gfx950. The trimmed-prefix budget is Arm A's top-k plus expand, timed in the same session. The probe does not time Arm A on the pad table, so the pad rows have no budget verdict.

| Arch | Case | 0 B | 16 KiB | 32 KiB | 64 KiB | top-k+expand budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gfx942 | p32k 0 B copy (unmodified in brackets) | 1799.7 / 1799.4 (1801.1 / 1800.5) | | | | |
| gfx942 | p128k pad | 5881.9 / 5923.5 | 6046.6 / 6077.2 | fail | fail | not measured |
| gfx942 | p256k pad | 11672.6 / 11658.3 | 11829.2 / 11799.4 | fail | fail | not measured |
| gfx942 | p128k trim | 4876.1 / 4917.2 | 4967.7 / 4904.4 | fail | fail | 2481.3 / 2486.6 |

The 0-byte copy ratio to unmodified is 0.9992 and 0.9994, inside the 5% faithfulness bar. Deltas against the same-repeat base for 16 KiB: `p128k` pad 164.7 / 153.7, `p256k` pad 156.6 / 141.1, `p128k` trim 91.5 / -12.9 µs. Scorer LDS before the candidate array is about 34.4 KB by the layout, so 16 KiB totals about 50.8 KB and 32 KiB about 67.2 KB against gfx942's 64 KiB. That is a layout estimate, not a compiler report.

### Kernel gate

Median microseconds per QSA layer per step, gfx950.

| Case | Fused | aiter#5996 | vllm#65188 | Faster arm | Margin | Pass |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| | | | | | | |

### Serving gate

| Input | Concurrency | TTFT fused | TTFT vllm#65188 | ITL fused | ITL vllm#65188 | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| | | | | | | |

GSM8K and GPQA Diamond:

### Decode table

| Rows | Context | Fused score | Fused merge | Current path | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| | | | | | |

## Decision record

Phase 0, gate 5: does not fire. The profile-only close is not taken. Phase 1 is next. The `p128k` share was read on the 65856-column pad, not the trimmed 32928-column prefix. The conclusion stands: gate 5 needs both shares under 0.15, and `p256k` is 0.334 on the 168-page table, which is the width a prefill step launches at `max_model_len`.

Outcome:

Gates met: Gate 5 was applied on the 2026-11-12 interleaved warm-2 medians and did not fire. No acceptance gate is met yet.

Margins: Arm A top-k-plus-expand share is 0.454 at `p128k` (1724.4 µs of 3797.3 µs) and 0.334 at `p256k` (1955.5 µs of 5861.4 µs). Gate 5 at `p128k` used that full 65856-column pad. A trimmed prefill step at 128K is 32928 columns, so 0.454 is not the trimmed-prefix share. `p256k` is the 168-page table, which trim leaves unchanged at `max_model_len`, and 0.334 is already above 0.15, so the gate does not fire on either reading. 1724.4 µs is not the trimmed-prefix budget. At `p256k` the faster arm is aiter#5996 at 5861.4 µs. Arm B's 66392.3 µs is the per-row scorer: the trimmed width makes 512-row chunks, and the MFMA tile starts at 768 rows.

gfx942 enablement:

Fallback used (yes or no, and why):

## Open questions

- Should decode rows of a mixed step share the prefill launch or keep a separate small launch? Phase 5 answers this by measurement.
- Which split count and candidate layout per context bucket? Phase 1 chose 4-byte candidates for 16 rows and the split counts in [Candidate state](#candidate-state). Phase 6 retunes a case that loses.
- On gfx942, 16 KiB of candidates fits beside the key tile and 32 KiB does not. Whether a 16 KiB state wins there is a Phase 6 measurement.

## Log

- 2026-11-12: Plan written from [design-spike.md](design-spike.md). On the gfx950 spike sessions, option 2 was eliminated by rule 2 and option 1 selected by rule 6. The spike measured the 64×64 logits scorer, so the candidate-state budget is re-measured on the 16-row indexer tile in Phase 1 before kernel work.
- 2026-11-12: Test environment now requires kernels to be compiled before the related work is queued with `gpuq`.
- 2026-11-12: Phase 0 paused. `op_tests/bench_qsa_indexer_select.py` is written, with arm A coded but not measured. Arm B is unavailable: the local vLLM checkout `6ad2e06c4a` has the Qwen4Exp AMD ops but not the vllm#65188 tile scorer, and commit `3c5e29ffcd` is not in it. Arm C (vllm#62589) is not in the tree either and was not checked upstream. The `--compile-only` job is still queued in gpuq (position 34) and has not run. No timing exists yet, so the Baseline table stays empty.
- 2026-11-12: Phase 0 compile run, gpuq GPU 4 (rocm-smi 7, bus 0000:f5:00.0, MI355X, `0x75a3`, gfx950). Idle at grant. The indexer and expand compiled for `p8k`, `p32k`, `p128k`, `p256k`, `two128k`, `four128k`, `mix3`, `mix16`, and `mix64`. The sparse GQA launch then failed with `DSLCompileError` at `fly.copy_atom_call_ssa` (buffer_copy 128, bf16) in `aiter/ops/flydsl/kernels/qsa/sparse_gqa.py` line 1066, the `fx.copy(g_copy, v_src, v_frag)` in the V gather. The log's last `compile launch` line before the error is `mix64`. The earlier GQA launches did not fail, so the error is specific to that launch and its cause is not diagnosed. Phase 0 cannot time until this compiles. The GQA leg is not part of the `T_step` split, but the plan times it alongside, so the bench stays blocked.
- 2026-11-12: vLLM arms are pinned. vllm#65188 is branch `rocm/qwen38-fn-wide-mono`; Phase 0 arm B is `3c5e29ffcd`, the MFMA prefill scorer, because PR head `0c1f099a42` reverts that prefill work. vllm#62589 is branch `qsa-amd-indexer-tiled`; Phase 0 arm C is the tile scorer at `e7a2acb5e3`. Serving stays on the #65188 head.
- 2026-11-12: Arm A compiles on the host with every GPU hidden (`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` empty, `AITER_AOT_REPLAY=1`, `GPU_ARCHS=gfx950`, `dry_run("gfx950", 256)`). The indexer compiled for all ten bench cases, including `mtp`. Expand is the Triton kernel `_expand_selected_groups`; Triton's `warmup` compiled it without a launch, ten calls, one specialization (`RATIO=4`, `BLOCK=128`), which was already in the cache from the earlier GPU compile. Sparse GQA still fails on `mix64` with the same `DSLCompileError` at `sparse_gqa.py` line 1066, so that failure does not need a device. The queued gpuq compile job was cancelled. A timed run shares these caches: importing aiter points `FLYDSL_RUNTIME_CACHE_DIR` at `aiter/jit/flydsl_cache` unless it is already set (`aiter/__init__.py`), and Triton uses `~/.triton/cache` when `TRITON_CACHE_DIR` is unset. The host dry run used 256 CUs, the full-chip gfx950 count. The split top-k bakes `get_cu_num()` into its specialization, so a GPU that reports another count recompiles that kernel. The one-workgroup top-k and the scorer do not. Arm A is still not timed. The bench can skip GQA with `--skip-gqa`.
- 2026-11-12: Arm C adapter is `op_tests/qsa_phase0_arm_vllm62589.py`. It loads only vllm#62589 commit `e7a2acb5e3`'s `qsa.py`. The PR was still open. The only C++ op is `top_k_per_row_decode`, same source as the installed build on ROCm. Host compile produced 16 Triton specializations; GPU runs compiled none. Reference session: gpuq 6, rocm-smi 5, bus 0000:95:00.0, MI355X, gfx950, live K pages filled. Row-set agreement with Arm A is 0.984 at `p8k` and 1.0 at `p32k` and `p128k` (one later session differs by one `p128k` row); differing blocks tie the 512th score. Warm `T_step` is 4417.7 µs at `p128k` and 6823.8 µs at `p256k`, top-k-plus-expand share 0.30 to 0.41. A second idle session matches within about 1%. `num_decode_rows` is ignored because that commit's API has no decode count. These are not the interleaved baseline, so the Arm C box stays open.
- 2026-11-12: Arm B adapter is `op_tests/qsa_phase0_arm_vllm65188.py`. It loads vllm#65188 commit `3c5e29ffcd` from materialized sources, not a checkout of `/workspaces/vllm`. The only C++ op on the path is `torch.ops._C.top_k_per_row_decode`, and its source matches the installed main build. Host compile (GPUs hidden, aiter `dry_run`) produced 9 Triton specializations; the GPU run compiled none. Sanity session: gpuq 7, rocm-smi 5, bus 0000:95:00.0, MI355X, gfx950. An earlier grant on bus 0000:65:00.0 reported 3% use and was discarded. Against Arm A, row-set agreement is 0.984 to 1.0 and block agreement is at least 0.99978; differing blocks are ties at the 512th score. On the bench's uninitialized K, `p128k` `T_step` is 11221 µs with all 16 chunks on the MFMA scorer. `p256k` never reaches that scorer: the trimmed width makes 512-row chunks, under the 768-row gate, and `T_score` is 64667 µs. Filling the live K pages with random values drops `p128k` top-k from 8185 µs to 1330 µs. The bench's K pool is `torch.empty`, so Arm A's Baseline top-k has the same problem. These are not the interleaved baseline, so the Arm B box stays open.
- 2026-11-12: Arm A timed, after the sparse GQA `mix64` compile was fixed and the host compile of all ten cases completed. gpuq 6, rocm-smi 6, bus 0000:e5:00.0, MI355X `0x75a3`, gfx950, 256 CUs. Idle at grant (use 0%, no other pids). A first grant on bus 0000:15:00.0 had other pids and exited 86; that session was discarded. Warm-2 medians are in the Baseline table. Sparse GQA ran and is not in `T_step`; its warm-2 time is 1677.2 µs on `p8k` and 3712.9 to 3735.5 µs on the 16384-row cases. The bench's gate-5 line is `profile_only_close=False`: the warm-2 top-k-plus-expand share is 0.803 at `p128k` and 0.765 at `p256k`. These are one chunk per case, so the cases checkbox and the gate-5 checkbox stay open. Warm-1 and the cold repeat sit within about 2% of warm-2 on `T_step`.
- 2026-11-12: Interleaved A/B/C session, `op_tests/bench_qsa_indexer_select_abc.py`. gpuq 4, rocm-smi 7, bus 0000:f5:00.0, MI355X `0x75a3`, gfx950. Idle at grant. A first grant on bus 0000:15:00.0 had other pids and was discarded. The 5460 live K pages were filled with random bf16. Arms alternated inside each repeat. Warm-2 medians are the Baseline table. On filled K, Arm A `T_step` is 3797.3 µs at `p128k` and 5861.4 µs at `p256k`, and the top-k-plus-expand share is 0.454 and 0.334, so gate 5 still does not fire. Arm B is faster than Arm A except at `p256k`, where it stays on the per-row scorer and `T_step` is 66392 µs. Arm C is slower than Arm A on every case in this session. Cold repeats stay within about 1% of warm-2. This is still one chunk per case, so the cases checkbox and gate 5 stay open.
- 2026-11-18: Phase 0 closed. A case is one scheduler step, so the Baseline table is the gate-5 input and the prompt is not re-summed. Gate 5 does not fire: Arm A warm-2 shares are 0.454 at `p128k` and 0.334 at `p256k`. The profile-only close is not taken. `p256k` top-k-plus-expand is 1955.5 µs on the 168-page table. The `p128k` 1724.4 µs budget is the untrimmed pad; Phase 1 keeps the 32 KiB candidate cap unless a trimmed-prefix timing (32928 columns) covers a larger array. Two- and four-prompt and mixed steps at 32K and 256K stay in the Phase 6 matrix. 1M stays out.
- 2026-11-18: Gate 5's `p128k` share is the 65856-column pad, not the trimmed 32928-column prefix. `p256k` is 0.334 on the width trim leaves at `max_model_len`, so the gate still does not fire.
- 2026-11-18: Superseded by the remeasure below. This run wrote every workgroup's keep-alive to one shared sink slot, which raced across workgroups, so its `p32k` match and `T_score` numbers are not used. Phase 1 candidate probe, gfx950, gpuq 0, rocm-smi 4, bus 0000:85:00.0, MI355X. Idle at grant (use 0%, no other pids). Wide cache span 4295165952 bytes. The spike copy is `aiter/ops/flydsl/kernels/qsa/indexer_select_lds_probe.py`; the production scorer is unchanged. Kernels were host-compiled with the GPUs hidden before the timed job. The 0-byte copy matched the unmodified scores on all 33 `p32k` chunks. Warm-2 `T_score` was 700.8 µs against 698.7 µs unmodified (ratio 1.003); warm-1 was 0.967. Both are inside 5%. Warm-2 slowdowns against the same-session unmodified scorer: `p128k` pad 281.1 / 926.0 / 3187.2 µs, `p256k` 416.7 / 1577.3 / 5819.7 µs, `p128k` trim 229.9 / 698.7 / 2338.2 µs, for 16 / 32 / 64 KiB. All three sizes launched. The trimmed Arm A budget is 1500.3 µs (`T_topk` 1375.3, `T_expand` 125.0). 16 KiB and 32 KiB stay under every budget, including 1955.5 µs at `p256k`. 64 KiB is over all three, so the 32 KiB cap stays. gfx942 and the layout choice are still open.
- 2026-11-18: The keep-alive store now writes one sink slot per workgroup, replacing the shared slot in the superseded run above. Remeasured on gpuq 0, rocm-smi 5, bus 0000:95:00.0, MI355X, gfx950, idle at grant. Warm-2 `p32k` ratio is 1.022 (718.8 µs against 703.4 µs). Warm-2 slowdowns: `p128k` pad 296.7 / 945.8 / 3214.5 µs, `p256k` 428.7 / 1600.7 / 5838.0 µs, `p128k` trim 240.0 / 704.6 / 2354.5 µs. The trimmed Arm A budget is 1499.4 µs. 16 KiB and 32 KiB stay under every budget. 64 KiB stays over, so the 32 KiB cap stays.
- 2026-11-18: Phase 1 on gfx942, `op_tests/spike_silotiger_1481_indexer_lds_probe.py`. `gpuq` is not available on this node, so nothing was queued. The user asked for the run without it. That departs from the locked gpuq rule, and the rule is unchanged in the plan. The run pinned `HIP_VISIBLE_DEVICES=7` to pick GPU 7, which is rocm-smi 7, bus 0000:cf:00.0, MI300X `0x74a1`, 304 CUs. The probe's own idle check passed (use 0%, no other pids). Host compile ran with every GPU hidden, `GPU_ARCHS=gfx942`, and `CU_NUM` 304, before the timed run. The compile needed `ld.lld` from `/tmp/rocm_env/llvm/bin` on `PATH`.
- 2026-11-18: The faithfulness check compared whole score buffers. The scorer writes only each row's visible prefix (columns below `row_lens`). The rest of the buffer keeps memory left by earlier allocations. On gfx942 that memory changed between launches, and two identical production runs differed in up to 4e8 values. A sentinel-prefilled buffer showed the written set is identical across runs. The check now compares the visible prefix only and fails if any visible score is unwritten. The gfx950 faithfulness record above used the old whole-buffer check, which can pass when the allocator returns the same memory each time. The re-run with the visible-prefix check is the next log entry.
- 2026-11-18: gfx942 fit. The 0-byte copy matches the unmodified scores on the visible prefix, ratio 0.9992 and 0.9994. 16 KiB launches and matches. 32 KiB and 64 KiB fail with `hipErrorIllegalState`. A failed launch leaves the HIP context in an error state, so every later launch in that process also fails. Those sizes were therefore run alone in fresh processes. The gfx942 candidate cap is 16 KiB per workgroup, and the gfx950 32 KiB cap does not carry over. 16 KiB holds 512 eight-byte candidates for 4 rows, or 512 four-byte candidates for 8 rows. The candidate layout choice is still open.
- 2026-11-18: gfx942 timing for 16 KiB, in the table above. Against the in-session trimmed Arm A budget (2481 to 2487 µs), the 16 KiB deltas are 91.5 and -12.9 µs. The probe does not time Arm A on the pad table on gfx942, so the pad rows have no verdict.
- 2026-11-18: gfx950 remeasure with the visible-prefix check, `op_tests/spike_silotiger_1481_indexer_lds_probe.py`. gpuq 0, rocm-smi 7, bus 0000:f5:00.0, MI355X, gfx950. Idle at grant (use 0%, no other pids). A first grant on bus 0000:15:00.0 had other pids and exited 86; that session was discarded. Kernels were host-compiled with the GPUs hidden before the timed job. The 0-byte copy matches the unmodified visible prefix on all 33 `p32k` chunks. Warm-2 `T_score` is 699.2 µs against 703.2 µs unmodified (ratio 0.994); warm-1 is 0.975. Both are inside 5%. `LAUNCH_OK` is true for 16, 32 and 64 KiB, each matching the same visible prefix, all in this process: 64 KiB launched, so no size needed a fresh process. Warm-2 slowdowns: `p128k` pad 299.6 / 950.8 / 3205.2 µs, `p256k` 429.1 / 1602.6 / 5839.7 µs, `p128k` trim 238.8 / 709.1 / 2348.5 µs, for 16 / 32 / 64 KiB. The trimmed Arm A budget is 1498.8 µs (`T_topk` 1375.1, `T_expand` 123.7). 16 KiB and 32 KiB stay under every budget, including 1955.5 µs at `p256k`. 64 KiB stays over, so the gfx950 32 KiB cap stays. The candidate layout choice is still open. This session's cache was a contiguous pool past 4 GiB, so the wide gather ran with dense strides. The strided-view remeasure is the next log entry.
- 2026-11-18: gfx950 remeasure on the acceptance cache. `_wide_indexer_cache` is now a per-layer view, page stride 262176, span 4295067584 bytes, and the live pages hold random bf16. `op_tests/spike_silotiger_1481_indexer_lds_probe.py`. gpuq device 0, rocm-smi 1, bus 0000:15:00.0, MI355X, gfx950. Idle at grant (use 0%, no other pids). Grants on 0000:95:00.0 (use 2%) and 0000:05:00.0 (other pids) exited 86 and were discarded. Kernels were host-compiled with the GPUs hidden before the timed job. The 0-byte copy matches the unmodified visible prefix on all 33 `p32k` chunks. Warm-2 `T_score` is 711.4 µs against 704.8 µs unmodified (ratio 1.009); warm-1 is 0.978. Both are inside 5%. `LAUNCH_OK` is true for 16, 32 and 64 KiB, all in this process. Warm-2 slowdowns: `p128k` pad 311.1 / 952.9 / 3214.6 µs, `p256k` 429.7 / 1590.8 / 5839.5 µs, `p128k` trim 242.2 / 714.8 / 2351.1 µs, for 16 / 32 / 64 KiB. In-session Arm A budgets: `p128k` pad 1721.2 µs (share 0.455), `p128k` trim 1499.5 µs (share 0.494), `p256k` 1908.6 µs (share 0.329). 16 KiB and 32 KiB stay under every budget. 64 KiB stays over, so the gfx950 32 KiB cap stays. The strided shares stay above 0.15, so gate 5 still does not fire and the baseline table was not re-run. The candidate layout choice is still open.
- 2026-11-18: Candidate layout chosen from the strided-cache probe. 4-byte candidates (16-bit order-preserving key and 16-bit column offset) for all 16 rows, 32 KiB of LDS. The tile itself is 34400 bytes, so the workgroup total is 67168 bytes. 64 KiB of candidates is over every gfx950 budget, and the 32 KiB dead-array slowdown stays under, with 317.8 µs to spare at `p256k`. One split at 8K, 32K and 128K (2352, 8232 and 32928 columns). Two splits of 32928 at 256K, because 65856 columns exceeds the 16-bit offset. gfx942 has 31136 bytes free beside the tile, so this state does not launch there and the fused path stays on the current indexer until Phase 6.
- 2026-11-18: Phase 2 oracle check and set-gate, gfx950, gpuq device 0, rocm-smi 2, bus 0000:65:00.0, MI355X, idle at grant. The fp32 oracle matches the fp64 top-512 on the seeded 8K (256 rows), 32K (128 rows), and padded-prefix cases. The largest score gap is 1.62e-6, and the tie bound is 1.7e-6. `qsa_indexer_select` is the entry the set-gate calls. Until the fused kernel replaces its body, that entry runs today's indexer and expand, and it does not clear the caller's score workspace. The set-gate covered a 168-page table, a missing page, a strided cache past 4 GiB, a 16-row tile across two requests, decode rows ahead of prefill rows, MTP rows, and a one-page emit (`n_columns` 392). All eight cases matched the oracle. gfx942 stays gated off.
- 2026-11-18: Phase 3 fused score and split-local top-512. The kernel is `aiter/ops/flydsl/kernels/qsa/indexer_fused_select.py`. A workgroup owns one 16-row tile and one column span from `fused_split_spans` (one span through 128K, two of 32928 at 256K). It keeps the bf16 MFMA and the 4-head ReLU sum, compares candidates with the fp32 score in the 32 KiB LDS array, and writes a packed 16-bit key plus 16-bit span offset. The first-level histogram is 256 bins of that key, updated on insert and eviction. Invisible, incomplete, and missing-page columns are not inserted. `qsa_indexer_select` uses this path when `n_columns > 512`; the one-page emit path is unchanged. The torch merge returns the kernel's columns for one span and re-ranks two spans in fp32. The Phase 2 set-gate passed on gfx950, gpuq device 0, rocm-smi 4, bus 0000:85:00.0, idle at grant (use 0%, no other pids). A grant on bus 0000:15:00.0 had use 2% and exited 86. All eight cases have `select err` 0. The gate's `select us` is the serial 512-slot scan on these small batches, recorded under [Fused select, Phase 3](#fused-select-phase-3), and is not a serving measurement. gfx942 stays gated off. The real merge is Phase 4.
- 2026-11-18: Phase 4 split-aware merge. The kernel is `aiter/ops/flydsl/kernels/qsa/indexer_fused_merge.py`. One workgroup per row sums the per-span 256-bin histograms, walks them from the high bin, and runs the second radix pass on the low byte of the one bin that straddles 512. Exact ties on that final key may resolve to either block. It writes `block_ids [M, 512]` and `token_ids [M, 2051]` (four tokens per block, the incomplete tail, `-1` padded) and zeros the histogram bins it read. The scorer still zeros the histogram before it fills it. `qsa_indexer_select` returns those token ids when `n_columns > 512` and no longer calls the Triton expand on that path. The one-page emit path is unchanged. Host compile covered the narrow and wide scorers and the merge (`dry_run("gfx950", 256)`, GPUs hidden). The Phase 2 set-gate passed on gfx950, gpuq device 0, rocm-smi 6, bus 0000:e5:00.0, idle at grant (use 0%, no other pids). All eight cases have `select err` 0, recorded under [Fused select, Phase 4](#fused-select-phase-4). The gate's contexts are one span, so this run does not exercise the second pass. gfx942 stays gated off.

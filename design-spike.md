# SILOTIGER-1481 design spike

Living plan for the pre-implementation probe that picks how the QSA indexer stops writing a full fp32 score buffer. Check a box when the step is done. Put measured numbers in [Results](#results), and dated notes in [Log](#log). Leave the prose in place when a step finishes.

The ticket is [SILOTIGER-1481-description.md](SILOTIGER-1481-description.md). This spike answers only the design choice in that ticket's "Suggested approach": whether fusion is the lever, and if it is, which mechanism wins on the locked device. It does not implement the fused kernel.

## Locked decisions

Change these only by editing this section and adding a log line. Probe code and the write-up follow this section as it stands.

### Test environment

- Interpreter: `/workspaces/aiter/flydsl_venv/bin/python` (Python 3.14.4, torch 2.12.0+rocm7.14.0).
- Any software that runs on a GPU (harness, probe, test, or bench) is always launched in `flydsl_venv` and through `gpuq` as specified in [`gpuq.md`](gpuq.md): set `GPUQ_USER` to the job owner and run `gpuq --num-gpus=1`. Do not start GPU work with the bare interpreter, and do not pin a device with `HIP_VISIBLE_DEVICES`. `gpuq` picks the GPU.
- Device: one GPU, assigned by `gpuq`. Record its index in [Results](#results).
- Observed when this plan was written: `rocm-smi` reports GPU 6 as AMD Instinct MI300X, device id `0x74a1`. Every number in this document is gfx942 evidence. The first harness run records the `get_gfx()` string next to that.
- Launch prefix for every timed command:

```bash
export GPUQ_USER="<owner>"
gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python
```

- Before each timed session, confirm the assigned GPU is idle (`rocm-smi -d <index> --showuse` shows no activity, and `rocm-smi --showpids` has no process on that GPU). One process holds the device for the whole session.
- Branch: `samaario/qsa-indexer-fused-select`.

The ticket's acceptance gate is MI355X (gfx950). This spike does not claim that gate. A later gfx950 run is a new session with its own environment lock, not a retarget of this one.

### Probe base

The kernel under test is the in-tree prefill path:

- Scorer: `qsa_paged_mqa_logits_flydsl` in `aiter/ops/flydsl/kernels/mqa_logits/qsa_flydsl_layout.py`. Tile is 64 query rows by 64 compressed groups, 4 heads, head dim 128, bf16 inputs, fp32 score store. Compressed page size is 16 (`PAGE_SIZE`), from a 64-token KV page and `compress_ratio` 4.
- Select: `qsa_select_paged_tokens` in `aiter/ops/qsa.py`, called with `allow_flydsl=True`. It caps the fp32 score workspace at `DEFAULT_LOGITS_WORKSPACE_BYTES` (128 MiB), chunks rows, then calls `top_k_per_row_prefill` and `qsa_expand_block_indices`.

`qsa_indexer_block_ids` from aiter#5996 (16×64 tile, compressed page 392) is not in this tree. This spike does not port it and does not quote its timings as if they were measured here. The LDS and tile numbers below are for the 64×64 scorer.

Current LDS struct in that kernel (`BN=64`, `HDIMS=128`, `BM=64`):

| Field | Bytes |
| --- | ---: |
| `k_tile`: 64 × (128 + 8) bf16 | 17408 |
| `row_valid_tag`: 64 int32 | 256 |
| `row_vis`: 64 int32 | 256 |
| `block_info`: 4 int32 | 16 |
| Total | 17936 |

gfx942 LDS is 64 KiB (65536 bytes), so about 46 KiB is free beside the key tile. One workgroup uses 17936 bytes, so three workgroups fit per CU. That occupancy is the control for probe A.

### Shapes

One step per case. Rows are the step's query rows. Context is the visible compressed-group count (`seq_len // 4`) for every row of a request, which is the wide end of a prefill chunk. Page table width is that column count divided by 16, with no extra padding. K cache is contiguous. `token_topk` is 2048, so the select keeps 512 blocks.

| Case | Requests × (rows, columns) | Chunks at 128 MiB |
| --- | --- | ---: |
| `s8k` | 1 × (8192, 2048) | 1 |
| `s32k` | 1 × (16384, 8192) | 4 |
| `s128k` | 1 × (16384, 32768) | 16 |
| `s256k` | 1 × (16384, 65536) | 32 |
| `two128k` | 2 × (8192, 32768) | 16 |
| `mix128k` | 1 × (16384, 32768) plus 64 decode rows at the same context | 17 |

`s128k` and `s256k` are the decision cases. The other four are context: short context, a second request, and a mixed step. A 1M case is out of this spike.

Score traffic and unique K for the decision cases, one request, 16384 rows:

| Case | fp32 score write | top-k reread | unique K |
| --- | ---: | ---: | ---: |
| `s128k` | 2 GiB | 2 GiB | 8 MiB |
| `s256k` | 4 GiB | 4 GiB | 16 MiB |

Unique K is one 256-byte bf16 vector per compressed group. It fits in the MI300X last-level cache (256 MiB) for every case above, including `two128k` (16 MiB). The hypothesis to test is that a second score pass is an L2 hit, and that the buffer's write plus reread is the traffic fusion would remove.

### Timing

- CUDA events around each stage, summed across chunks of that step. One number per stage per step: `T_score`, `T_topk`, `T_expand`. `T_step` is the sum.
- 5 warmup steps, then 10 timed steps. Report the median. Same protocol for both passes of probe B, with the two passes launched back to back inside one timed step so pass 2 sees pass 1's cache.
- Control times come from the unmodified scorer. A spike copy is faithful only after its zero-dummy variant matches the unmodified logits on `s32k` and its median scorer time is within 5% of the unmodified median.

### Stop and elimination rules

Apply in order. Fill [Decision](#decision) when a rule fires, and skip the later phases.

1. **Stop.** On both `s128k` and `s256k`, `(T_topk + T_expand) / T_step < 0.15`. Fusion is the wrong lever on this device. Later work goes to scorer efficiency.
2. **Option 2 out.** On both decision cases, `T_pass1 + T_pass2 > T_step`. The second score costs more than today's top-k plus expand, before any histogram cost.
3. **Option 1 out.** The 32 KiB dummy (8 rows × 512 candidates × 8 bytes), added beside the existing 17936-byte struct, slows `T_score` by more than `T_topk + T_expand` on either decision case. The occupancy loss spends the fusion budget. A 64 KiB dummy that fails to launch is a recorded fit result, not a timing result: 16 rows of candidates do not fit beside the key tile on this GPU.
4. **Both out.** Choose the 16-bit score-buffer fallback for the follow-up implementation: keep one score pass, store order-preserving 16-bit keys, halve the buffer, and rescore the boundary bucket in fp32.
5. **Both still in, within 10% on `T_step`.** Run the histogram probe. It is part of option 2's cost. If it pushes option 2 over `T_step`, option 2 is out. If the two full estimates stay within 10% after that, record "too close on gfx942" and do not invent a gfx950 winner.
6. **Otherwise** the cheaper surviving estimate wins. State the margin in the decision.

Option 1's estimate is the slowed scorer plus a merge treated as no larger than today's `T_topk` (an upper bound: the merge sees a handful of 512-candidate lists, not `columns` scores). Option 2's estimate is `T_pass1 + T_pass2` plus the histogram-probe delta when that probe ran.

## Phases

### 0. Harness

- [x] Add `op_tests/spike_silotiger_1481_select_probe.py`. It builds the six shapes, calls `qsa_select_paged_tokens(..., allow_flydsl=True)`, and times score, top-k, and expand separately by wrapping those three calls. Chunk count and score bytes written and read are printed per case.
- [x] Correctness smoke on `s8k` and `s32k`: the selected block set matches a fp32 torch top-512 over the same window, ties at the 512th score allowed. This guards the harness, and it is not the ticket's fp64 set gate.
- [x] Confirm `get_gfx()` and the idle check, and write both into [Results](#results).

### 1. Baseline split

- [x] Run all six cases with the unmodified path.
- [x] Fill the baseline table.
- [x] Apply rule 1. If it fires, check Decision and stop.

### 2. Probe A — LDS occupancy for local top-512

Skip if rule 1 fired.

- [x] Add a spike copy of the scorer, `aiter/ops/flydsl/kernels/mqa_logits/qsa_flydsl_layout_lds_probe.py`, with one extra LDS array of a compile-time byte size. Do not change `qsa_flydsl_layout.py`.
- [x] Faithfulness: dummy size 0 matches unmodified logits on `s32k`, and its median `T_score` is within 5% of the phase-1 median.
- [x] Time dummy 32768 bytes on `s128k` and `s256k`.
- [x] Attempt dummy 65536 bytes on `s128k`. Record launch success or the compiler/loader error. Do not retry with a smaller tile in this spike.
- [x] Apply rule 3.

### 3. Probe B — second score pass

Skip if rule 1 fired. Run even when option 1 is already out, so the fallback choice has both numbers.

- [x] On all six cases, launch the unmodified scorer twice back to back. Record `T_pass1` and `T_pass2` separately.
- [x] Apply rule 2.
- [x] Histogram probe, only if rules 2 and 3 both left their option alive and rule 5 says the estimates are within 10%. In the spike scorer, increment a coarsened LDS histogram (256 int32 bins for 16 rows, 16 KiB) from the existing score loop on one wave's rows, and store the histograms out. Delta is `(T_hist - T_score)` on `s128k` and `s256k`. Add that delta to option 2's estimate and apply rule 5 again.

### 4. Decision

- [x] Write the outcome, the rule that fired, and the margins into [Decision](#decision).
- [x] Log the date.

## Results

### Environment confirmation

| Item | Value |
| --- | --- |
| Phase 0 date | 2026-11-11 |
| Phase 0 `get_gfx()` | gfx950 (`get_gfx_runtime()` agrees) |
| Phase 0 GPU | gpuq 3, rocm-smi 1, bus 0000:15:00.0, AMD Instinct MI355X, device id `0x75a3` |
| Phase 0 idle | Use was 0%. PID 4144877 already held 1762750464 bytes of VRAM on that GPU (CU occupancy 1). gpuq showed no lease. |
| Baseline date | 2026-11-12 |
| Baseline `get_gfx()` | gfx950 (`get_gfx_runtime()` agrees) |
| Baseline GPU | gpuq 2, rocm-smi 2, bus 0000:65:00.0, AMD Instinct MI355X, device id `0x75a3` |
| Baseline idle | Yes. Use was 0%, `rocm-smi --showpids` had no process on that GPU, and gpuq showed no lease. |
| Probe A date | 2026-11-12 |
| Probe A `get_gfx()` | gfx950 (`get_gfx_runtime()` agrees) |
| Probe A GPU | gpuq 4, rocm-smi 7, bus 0000:f5:00.0, AMD Instinct MI355X, device id `0x75a3` |
| Probe A idle | Yes. Use was 0%, `rocm-smi --showpids` had no process on that GPU, and gpuq showed no lease. The assigned bus was checked before the session. |
| Probe B date | 2026-11-12 |
| Probe B `get_gfx()` | gfx950 (`get_gfx_runtime()` agrees) |
| Probe B GPU | gpuq 6, rocm-smi 6, bus 0000:e5:00.0, AMD Instinct MI355X, device id `0x75a3` |
| Probe B idle | Yes. Use was 0%, `rocm-smi --showpids` had no process on that GPU, and gpuq showed no lease. The assigned bus was checked before the session. |

Phase-0 smoke on its device passed for `s8k` (1 chunk, 67108864 score bytes written and read) and `s32k` (4 chunks, 536870912 score bytes written and read). The baseline below is a later session: 5 warmup steps, then the median of 10, on the unmodified logits chunk loop. `gpuq` assigned a gfx950 MI355X. The 2026-11-05 lock names gfx942. Every timed session on this node is gfx950. Rule 1 does not fire. The outcome of the rules on these sessions is in [Decision](#decision).

### Baseline

Median microseconds per step. Bytes are the fp32 score write, equal to the top-k reread.

| Case | Chunks | Score bytes | `T_score` | `T_topk` | `T_expand` | `T_step` | top-k+expand share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s8k` | 1 | 64 MiB | 77.1 | 61.7 | 64.5 | 203.2 | 0.621 |
| `s32k` | 4 | 512 MiB | 394.1 | 411.4 | 132.0 | 937.5 | 0.580 |
| `s128k` | 16 | 2 GiB | 1461.4 | 809.3 | 182.7 | 2453.5 | 0.404 |
| `s256k` | 32 | 4 GiB | 2972.5 | 1539.8 | 241.5 | 4753.7 | 0.375 |
| `two128k` | 16 | 2 GiB | 1468.2 | 828.5 | 189.2 | 2485.9 | 0.409 |
| `mix128k` | 17 | 2 GiB + 8 MiB | 1829.6 | 820.3 | 197.5 | 2847.4 | 0.357 |

`T_step` is the sum of the three stage medians. The share is `(T_topk + T_expand) / T_step`. Rule 1 needs that share below 0.15 on both `s128k` and `s256k`. The shares are 0.404 and 0.375, so rule 1 does not fire. Every chunk of `s8k`, `s32k`, `s128k`, `s256k`, and `two128k` entered `qsa_paged_mqa_logits_flydsl`. `mix128k` entered it on 16 of 17 chunks. Its last chunk is 64 rows, and the unmodified select uses FlyDSL only when the chunk has more than 256 rows, so that chunk ran the Triton scorer. Its time is inside `T_score`.

### Probe A

Median `T_score`, microseconds. "fail" means the kernel did not launch.

| Case | dummy 0 | dummy 32 KiB | dummy 64 KiB |
| --- | ---: | ---: | ---: |
| `s32k` (faithfulness) | 384.4 | | |
| `s128k` | | 1573.4 | 2668.4 |
| `s256k` | | 3262.1 | |

Dummy-0 logit check on `s32k`: logits, visible counts, and row starts match the unmodified scorer in the same session. Median `T_score` is 384.4 us, against the phase-1 median 394.1 us (ratio 0.975) and the same-session unmodified median 398.3 us (ratio 0.965). Both ratios are inside 5%, so the copy is faithful and rule 3 applies.

Same-session unmodified `T_score` (the delta baseline, timed immediately before the matching dummy): `s32k` 398.3 us, `s128k` 1426.1 us, `s256k` 2904.7 us. The 32 KiB dummy adds 147.3 us on `s128k` and 357.4 us on `s256k`. Phase-1 `T_topk + T_expand` is 992.0 us and 1781.3 us. Neither delta exceeds its budget, so rule 3 does not fire. Option 1 stays in, and phase 3 still runs.

The 64 KiB dummy launched on this gfx950 GPU and its median is the table cell. gfx950 LDS is 160 KiB, so 17936 + 65536 = 83472 bytes fits one workgroup. The plan's fit failure is the gfx942 case, where LDS is 64 KiB. This spike did not retry a smaller tile. Rule 3 uses the 32 KiB deltas only.

### Probe B

Median microseconds, unmodified scorer, two back-to-back passes.

| Case | `T_pass1` | `T_pass2` | `T_pass1 + T_pass2` | phase-1 `T_step` |
| --- | ---: | ---: | ---: | ---: |
| `s8k` | 74.1 | 55.5 | 129.6 | 203.2 |
| `s32k` | 393.8 | 369.6 | 763.4 | 937.5 |
| `s128k` | 1426.4 | 1407.3 | 2833.7 | 2453.5 |
| `s256k` | 2907.6 | 2875.8 | 5783.5 | 4753.7 |
| `two128k` | 1432.5 | 1405.9 | 2838.4 | 2485.9 |
| `mix128k` | 1817.0 | 1794.2 | 3611.2 | 2847.4 |

Each timed step queues pass 1 across every chunk, then pass 2 across the same chunks, with no sync between the passes. `T_pass1` and `T_pass2` are medians of those chunk sums. The sum column is that pair added before the 1-decimal display, so `s256k` shows 5783.5 where the two displayed medians add to 5783.4. The scorer is `qsa_paged_mqa_logits(..., allow_flydsl=True)`, the same chunk loop as phase 1, including the Triton chunk inside `mix128k`.

Rule 2 needs `T_pass1 + T_pass2 > T_step` on both decision cases. `s128k` is over by 380.2 us (2833.7 versus 2453.5). `s256k` is over by 1029.8 us (5783.5 versus 4753.7). Rule 2 fires on this gfx950 session: option 2 is out before any histogram cost. Pass 2 is only slightly faster than pass 1 (1407.3 versus 1426.4 on `s128k`, 2875.8 versus 2907.6 on `s256k`).

The histogram probe did not run. Rule 3 left option 1 alive, but rule 2 did not leave option 2 alive. The pre-histogram estimates are also outside 10% of `T_step`: option 1 (probe-A slowed score plus phase-1 `T_topk`) is 2382.7 us on `s128k` and 4801.9 us on `s256k`, against option 2's two-pass sums above. The gaps are 0.184 and 0.206 of phase-1 `T_step`. [Decision](#decision) records the outcome. These sessions are gfx950 and leave the 2026-11-05 gfx942 lock as written.

Histogram delta (`T_hist - T_score`), if the probe ran:

| Case | delta |
| --- | ---: |
| `s128k` | |
| `s256k` | |

## Decision

Outcome: Option 1. On the gfx950 sessions measured here, the follow-up is a fused local top-512 plus a merge bounded by today's `T_topk`. Option 2 is eliminated. This is the rule result for these MI355X sessions. The 2026-11-05 lock still names gfx942, and this write-up leaves that lock in place. It also leaves the ticket's MI355X serving gate untouched.

Rule: Rule 2 fires, so option 2 is out. Rule 1 does not fire. Rule 3 does not fire, so option 1 stays in. Rule 4 needs both options out. Rule 5 needs both still in. Rule 6 selects the only surviving estimate.

Margins: Two unmodified score passes exceed phase-1 `T_step` by 380.2 us on `s128k` (2833.7 versus 2453.5, 15.5%) and by 1029.8 us on `s256k` (5783.5 versus 4753.7, 21.7%). Option 1's upper bound, the probe-A 32 KiB scorer plus phase-1 `T_topk`, is 2382.7 us on `s128k` (70.8 us under `T_step`) and 4801.9 us on `s256k` (48.2 us over `T_step`). That bound is the merge treated as no larger than today's top-k. Against the eliminated two-pass sums, the same bound is lower by 451.0 us on `s128k` and 981.6 us on `s256k`, which is 0.184 and 0.206 of phase-1 `T_step`. The 32 KiB occupancy deltas that kept option 1 alive are 147.3 us against a 992.0 us budget on `s128k`, and 357.4 us against a 1781.3 us budget on `s256k`.

Follow-up (one line; the fused kernel is a later change): Implement option 1 on the in-tree 64×64 scorer: fused local top-512 and a split-aware merge, in a later change.

## Out of scope

- Implementing option 1, option 2, or the 16-bit fallback.
- The ticket's serving gate: TTFT, ITL, GSM8K, GPQA, vllm#65188, and the #5996 head-to-head.
- A 4 GiB-strided cache view, page tables padded to `max_model_len` 262144, and the page-392 layout.
- A pure-decode sweep and any CUDA-graph capture.
- gfx950, and any change to the production scorer's numerics or launch.

## Log

- 2026-11-05: Plan written. No probe has been run. GPU 6 locked as the MI300X above; probe base locked to the in-tree 64×64 scorer because `qsa_indexer_block_ids` is not in the tree.
- 2026-11-11: Test environment now requires all GPU software to run in `flydsl_venv` through `gpuq` per `gpuq.md`. The `HIP_VISIBLE_DEVICES=6` pin is dropped; `gpuq` assigns the GPU, and the MI300X observation above still stands for GPU 6.
- 2026-11-11: Phase 0 harness is `op_tests/spike_silotiger_1481_select_probe.py`. `qsa_indexer_block_ids` is in this tree and `qsa_select_paged_tokens` takes it for these shapes, so the harness makes that gate reject and times the logits chunk loop (score, top-k, expand). `mix128k` is 64 one-row decode requests, then one 16384-row prefill, each with its own 32768-column cache (unique K 545259520 bytes). Smoke passed on `s8k` and `s32k`. `get_gfx()` is gfx950 on gpuq GPU 3 (rocm-smi GPU 1, bus 0000:15:00.0, MI355X, `0x75a3`). That GPU already had PID 4144877 resident. No phase-1 median was recorded.
- 2026-11-12: Phase 1 baseline, 5 warmup and median of 10, on gpuq GPU 2 (rocm-smi GPU 2, bus 0000:65:00.0, MI355X, `0x75a3`, gfx950). That GPU was idle before the session. `s128k` top-k+expand share 0.404, `s256k` share 0.375. Rule 1 does not fire. Phases 2 and 3 stay open. `mix128k` scored 16 FlyDSL chunks and one 64-row Triton chunk.
- 2026-11-12: Phase 2 probe A, 5 warmup and median of 10, on gpuq GPU 4 (rocm-smi GPU 7, bus 0000:f5:00.0, MI355X, `0x75a3`, gfx950). That GPU was idle before the session. Dummy 0 matches unmodified `s32k` logits and is within 5% of both the phase-1 and same-session scorer medians. The 32 KiB dummy adds 147.3 us on `s128k` and 357.4 us on `s256k`, both under the phase-1 top-k+expand budgets (992.0 us and 1781.3 us). Rule 3 does not fire. The 64 KiB dummy launched; gfx950 LDS is 160 KiB. Phase 3 stays open. `qsa_flydsl_layout.py` is unchanged.
- 2026-11-12: Phase 3 probe B, 5 warmup and median of 10, on gpuq GPU 6 (rocm-smi GPU 6, bus 0000:e5:00.0, MI355X, `0x75a3`, gfx950). That GPU was idle before the session. Two unmodified score passes sum to 2833.7 us on `s128k` and 5783.5 us on `s256k`, both above phase-1 `T_step` (2453.5 us and 4753.7 us). Rule 2 fires. The histogram probe did not run. Decision stays empty under the gfx942 lock.
- 2026-11-12: Phase 4 decision. On the gfx950 sessions, rule 2 eliminates option 2 and rule 6 selects option 1. The upper bound is 2382.7 us on `s128k` and 4801.9 us on `s256k`. The gfx942 lock and the ticket's serving gate stay as written.

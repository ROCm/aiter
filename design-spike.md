# SILOTIGER-1481 design spike

Living plan for the pre-implementation probe that picks how the QSA indexer stops writing a full fp32 score buffer. Check a box when the step is done. Put measured numbers in [Results](#results), and dated notes in [Log](#log). Leave the prose in place when a step finishes.

The ticket is [SILOTIGER-1481-description.md](SILOTIGER-1481-description.md). This spike answers only the design choice in that ticket's "Suggested approach": whether fusion is the lever, and if it is, which mechanism wins on the locked device. It does not implement the fused kernel.

## Locked decisions

Change these only by editing this section and adding a log line. Probe code and the write-up follow this section as it stands.

### Test environment

- Interpreter: `/workspaces/aiter/flydsl_venv/bin/python` (Python 3.14.4, torch 2.12.0+rocm7.14.0).
- Device: physical GPU 6, selected only by `HIP_VISIBLE_DEVICES=6`.
- Observed when this plan was written: `rocm-smi` reports GPU 6 as AMD Instinct MI300X, device id `0x74a1`. Every number in this document is gfx942 evidence. The first harness run records the `get_gfx()` string next to that.
- Launch prefix for every timed command:

```bash
HIP_VISIBLE_DEVICES=6 /workspaces/aiter/flydsl_venv/bin/python
```

- Before each timed session, confirm GPU 6 is idle (`rocm-smi -d 6 --showuse` shows no activity, and `rocm-smi --showpids` has no process on GPU 6). One process holds the device for the whole session.
- Branch: `samaario/silotiger-1481-qsa-indexer-fused-select`.

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

- [ ] Add `op_tests/spike_silotiger_1481_select_probe.py`. It builds the six shapes, calls `qsa_select_paged_tokens(..., allow_flydsl=True)`, and times score, top-k, and expand separately by wrapping those three calls. Chunk count and score bytes written and read are printed per case.
- [ ] Correctness smoke on `s8k` and `s32k`: the selected block set matches a fp32 torch top-512 over the same window, ties at the 512th score allowed. This guards the harness, and it is not the ticket's fp64 set gate.
- [ ] Confirm `get_gfx()` and the idle check, and write both into [Results](#results).

### 1. Baseline split

- [ ] Run all six cases with the unmodified path.
- [ ] Fill the baseline table.
- [ ] Apply rule 1. If it fires, check Decision and stop.

### 2. Probe A — LDS occupancy for local top-512

Skip if rule 1 fired.

- [ ] Add a spike copy of the scorer, `aiter/ops/flydsl/kernels/mqa_logits/qsa_flydsl_layout_lds_probe.py`, with one extra LDS array of a compile-time byte size. Do not change `qsa_flydsl_layout.py`.
- [ ] Faithfulness: dummy size 0 matches unmodified logits on `s32k`, and its median `T_score` is within 5% of the phase-1 median.
- [ ] Time dummy 32768 bytes on `s128k` and `s256k`.
- [ ] Attempt dummy 65536 bytes on `s128k`. Record launch success or the compiler/loader error. Do not retry with a smaller tile in this spike.
- [ ] Apply rule 3.

### 3. Probe B — second score pass

Skip if rule 1 fired. Run even when option 1 is already out, so the fallback choice has both numbers.

- [ ] On all six cases, launch the unmodified scorer twice back to back. Record `T_pass1` and `T_pass2` separately.
- [ ] Apply rule 2.
- [ ] Histogram probe, only if rules 2 and 3 both left their option alive and rule 5 says the estimates are within 10%. In the spike scorer, increment a coarsened LDS histogram (256 int32 bins for 16 rows, 16 KiB) from the existing score loop on one wave's rows, and store the histograms out. Delta is `(T_hist - T_score)` on `s128k` and `s256k`. Add that delta to option 2's estimate and apply rule 5 again.

### 4. Decision

- [ ] Write the outcome, the rule that fired, and the margins into [Decision](#decision).
- [ ] Log the date.

## Results

### Environment confirmation

| Item | Value |
| --- | --- |
| Date | |
| `get_gfx()` | |
| GPU 6 idle before the run | |

### Baseline

Median microseconds per step. Bytes are the fp32 score write, equal to the top-k reread.

| Case | Chunks | Score bytes | `T_score` | `T_topk` | `T_expand` | `T_step` | top-k+expand share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s8k` | 1 | | | | | | |
| `s32k` | 4 | | | | | | |
| `s128k` | 16 | 2 GiB | | | | | |
| `s256k` | 32 | 4 GiB | | | | | |
| `two128k` | 16 | | | | | | |
| `mix128k` | 17 | | | | | | |

### Probe A

Median `T_score`, microseconds. "fail" means the kernel did not launch.

| Case | dummy 0 | dummy 32 KiB | dummy 64 KiB |
| --- | ---: | ---: | ---: |
| `s32k` (faithfulness) | | | |
| `s128k` | | | |
| `s256k` | | | |

Dummy-0 logit check on `s32k`:

### Probe B

Median microseconds, unmodified scorer, two back-to-back passes.

| Case | `T_pass1` | `T_pass2` | `T_pass1 + T_pass2` | phase-1 `T_step` |
| --- | ---: | ---: | ---: | ---: |
| `s8k` | | | | |
| `s32k` | | | | |
| `s128k` | | | | |
| `s256k` | | | | |
| `two128k` | | | | |
| `mix128k` | | | | |

Histogram delta (`T_hist - T_score`), if the probe ran:

| Case | delta |
| --- | ---: |
| `s128k` | |
| `s256k` | |

## Decision

Outcome:

Rule:

Margins:

Follow-up (one line; the fused kernel is a later change):

## Out of scope

- Implementing option 1, option 2, or the 16-bit fallback.
- The ticket's serving gate: TTFT, ITL, GSM8K, GPQA, vllm#65188, and the #5996 head-to-head.
- A 4 GiB-strided cache view, page tables padded to `max_model_len` 262144, and the page-392 layout.
- A pure-decode sweep and any CUDA-graph capture.
- gfx950, and any change to the production scorer's numerics or launch.

## Log

- 2026-11-05: Plan written. No probe has been run. GPU 6 locked as the MI300X above; probe base locked to the in-tree 64×64 scorer because `qsa_indexer_block_ids` is not in the tree.

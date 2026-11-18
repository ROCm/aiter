## Why
Follow-up to [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047) (FlyDSL QSA indexer scorer and sparse GQA: ROCm/aiter#5996, gfx942 follow-up ROCm/aiter#6083).

The QSA indexer scores every complete 4-token block of the context, keeps the top 512 blocks and expands them to at most 2051 token ids. At long context it dominates the QSA layer. A rocprofv3 profile of the #5996 indexer (MI355X, at an earlier #5996 head `f8d72e40b`) put it at 55% of indexer-plus-sparse-GQA kernel time for single-request prefill at 128K, 89% at 1M, and 87% for a 4-request prefill at 128K (TP2 shapes). The indexer is replicated on every TP rank, so its cost does not shrink with TP.

Every AMD path today writes an fp32 score buffer `[rows, columns]`, then runs a separate per-row top-k over it and a separate expand:

* vLLM main's AMD `qsa_select_paged_tokens` caps the buffer at 128 MiB (`_LOGITS_WORKSPACE_BYTES`) and loops over row chunks. At the padded width of 65,856 columns (`max_model_len` 262144) a chunk holds 509 rows, so a 16384-token prefill step is 33 sequential score, top-k and expand rounds per QSA layer, each re-reading that request's compressed keys.
* vllm#65188 trims the score width to `max_seq_len` on eager steps. vllm#59437, which wires aiter#5996 into vLLM behind the same 128 MiB chunk loop, trims the page table to the pages `max_seq_len` reaches on every step with prefills. Either way a 128K context gets about 1024 rows per chunk (16 rounds per 16384-token step), but a 1M context gets 128 rows per chunk (128 rounds).

The #5996 indexer, with multi-row tiling, is about 2x faster than vllm#62589's Triton tile scorer on 8K prompts, but only at parity on a 16384-row chunk at 98K context (3469 vs 3499 µs per call). The lead closes as context grows, which is where the score buffer and the per-row top-k grow.

This ticket targets **prefill and mixed prefill/decode steps**. Pure decode runs in full CUDA graphs and is handled by specialized decode kernels (vllm#65188 `--monokernel`). Measure decode here only to inform those kernels.

## Kernel family and operation
Qwen3.8-Flash-Next QSA lightning indexer (`indexer_n_heads=4`, `indexer_head_dim=128`, `indexer_compress_ratio=4`, `indexer_budget=2048`; 12 of 48 layers). Config: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json . Math: Qwen3.8-Flash-Next tech report, section 2.1.

For query row `i` at absolute position `p_i` in request `r`:

1. Visible blocks: `n_i = min((p_i + 1) // 4, seq_len[r] // 4)`.
2. Score of block `b < n_i`: `s(i, b) = scale * sum_h ReLU(dot(q[i, h], kbar[r, b]))`. BF16 inputs, fp32 accumulation. `scale = 1/sqrt(128)` does not change the selection.
3. Select: if `n_i <= 512`, all visible blocks; otherwise the 512 highest scores. Output order and the pick among exact ties at the 512th score are free. Ordered or stable output is not required.
4. Expand: block `b` gives tokens `4b .. 4b + 3`, then the incomplete tail `4 * ((p_i + 1) // 4) .. p_i` (0 to 3 tokens). At most 2051 token ids, `-1` padded.

vLLM flattens a step's tokens into rows: decode rows first (`num_decode_tokens`), then prefill rows, each request's rows contiguous. A mixed step is decode rows (1 per request, 4 with MTP) plus one or more prefill chunks. `max_num_batched_tokens` defaults to 16384.

## Workload phase and regime
**Primary: prefill and mixed steps.** These run outside CUDA graphs: `vllm::qwen4_exp_qsa_with_output` and `vllm::qwen4_exp_qsa_prepare_with_output` are piecewise splitting ops (`vllm/config/compilation.py`). The host therefore has `num_decode_tokens`, `num_prefills`, `max_query_len` and `max_seq_len` for each step, and may pick a kernel variant per step by row count and context length. Rows run from hundreds to 16384, and many rows of one request share the same compressed keys. Key reuse across rows (multi-row MFMA tiles) is what makes the current scorers fast, and any fused design has to keep it.

**Secondary, measure only: pure decode.** `QSAIndexerMetadataBuilder` declares `UNIFORM_BATCH` graph support, and vLLM keys decode graphs only on token and request counts. The host cannot dispatch by context length there; a decode variant must choose per row on the GPU.

**Hardware:** MI355X (gfx950) first. MI300X (gfx942) measured; enable only where it wins.

## Input/output contract (draft)
The starting point is `qsa_indexer_block_ids` in `aiter/ops/flydsl/qsa_paged.py` (aiter#5996). A fused entry could look like this; names and argument split are a proposal:

```python
def qsa_indexer_select(
    q, k_cache, page_table, token_to_req, query_positions, context_lens,
    *, num_decode_rows, max_seq_len, block_ids=None, token_ids=None, workspace=None,
):
    # q: bf16 [M, 4, 128] normed, RoPE'd indexer queries; M = all tokens of the step
    # k_cache: bf16 [num_pages, 392, 1, 128] compressed index keys (1568-token main page / 4);
    #   in vLLM a strided per-layer view of one pool whose span exceeds 4 GiB
    # page_table: int32 [num_reqs, pages], padded to max_model_len (168 pages = 65,856 columns at 262144)
    # token_to_req: int32 [M]; query_positions: int32 [M] absolute positions
    # context_lens: int32 [num_reqs] sequence lengths including this step's tokens
    # num_decode_rows, max_seq_len: host ints from the step metadata (eager path only)
    # block_ids: int32 [M, 512] out; unique visible block ids per row, any order, -1 padded
    # token_ids: int32 [M, 2051] out; expanded ids and tail, -1 padded (what sparse GQA reads)
    # workspace: caller-owned, sized once from max_num_batched_tokens and max_model_len;
    #   reusable across calls without a memset
```

## Correctness reference
* `op_tests/qsa_ref.py` `qsa_oracle` in aiter#5996: fp32 block-causal ReLU-sum scores, exact top-512, expand and tail. Validate it against an fp64 rescore.
* Gate per row: set equality with the oracle, except blocks whose scores tie within the fp32 error bound at the 512th score. vllm#65188 commit `3c5e29ffcd` uses the same form: at least 99.9% block agreement with fp64, every differing block within twice the bound of the 512th score.
* Do not use another production kernel as the oracle.

## Related code paths
* vLLM AMD selection (production): `vllm/models/qwen4_exp/amd/ops/qsa.py` `qsa_select_paged_tokens` calls `qsa_mqa_paged` (per-row Triton; in vllm#65188 also the 64-by-64 Triton MFMA tile scorer `qsa_mfma_scorer.py` for prefill chunks of at least 768 rows), then `ops.top_k_per_row_decode` (HIP), then `expand_qsa_block_indices_cuda`.
* aiter#5996 (open): `qsa_indexer_block_ids` in `aiter/ops/flydsl/qsa_paged.py`, kernels in `aiter/ops/flydsl/kernels/qsa/indexer_select.py`. The multi-row scorer is 16 rows by 64 columns with four column tiles per workgroup, and folds 4 rows by 4 heads onto the MFMA M axis for MTP. It writes fp32 scores, then selects with the aiter FlyDSL decode radix (`flydsl_top_k_per_row_decode`) for prefill or `topk_select` for wide decode. vLLM wiring: vllm#59437 (`--attention-config.qwen4_exp_qsa_backend flydsl`).
* aiter#5525 (open draft): fused score and local top-k for the GLM DSA indexer (`aiter/ops/flydsl/kernels/mqa_logits/fp8_paged_mqa_local_topk.py`) with a split-aware merge (`aiter/ops/flydsl/split_topk_merge.py`). The scorer fills the merge's first radix histogram while scoring, the merge starts at the second pass, and the merge workspace zeroes itself, so graph replay needs no memset. It scores one query row per workgroup (no key reuse across rows) and is decode-only. [SILOTIGER-1136](https://amd.atlassian.net/browse/SILOTIGER-1136) is the open DSA prefill measurement.
* Top-k kernels: aiter `top_k_per_row_decode` and `topk_select` (including the `sampled` backend, aiter#5686 / #7014), and the multi-workgroup adaptive decode top-k aiter#5677 ([SILOTIGER-699](https://amd.atlassian.net/browse/SILOTIGER-699), [SILOTIGER-1414](https://amd.atlassian.net/browse/SILOTIGER-1414)).
* Other QSA scorers: aiter#6515 (FlyDSL prefill MFMA logits, page 64, merged), aiter `aiter/ops/qsa.py` `qsa_select_paged_tokens` (logits, then `top_k_per_row_prefill`), vllm#62589 (Triton tile scorer, open).
* Decode reference for later: vllm#65188 `vllm/models/qwen4_exp/amd/specialized_decode/wide/indexer.py` and `select.py` (in-kernel score and radix select, take-all below 512 blocks, 64 score workgroups per row up to 4 rows).

## Baseline and competitor evidence
MI355X, Qwen3.8-Flash-Next TP2 shapes. None of these is a matched comparison of all current paths.

| Source | What | Numbers |
| --- | --- | --- |
| aiter#5996 comment by mjkvaak-amd, 2026-10-22 | score, top-k and expand per call; table padded to 65,856 columns; cache view over 4 GiB; GPU time | #5996 indexer vs vllm#62589 Triton, µs: one 8192 prompt 381 vs 749; 8192 prompt and 3 decode rows 384 vs 858; two 8192 prompts 782 vs 1714; 16384-row chunk at 60K 2336 vs 2693; same with 3 decode rows 2366 vs 2922; 16384-row chunk at 98K 3469 vs 3499 |
| aiter#5996 comment by johannes-graner, 2026-10-09 (head `f8d72e40b`, before the 16x64 tiling) | rocprofv3, indexer scorer only, M=2048 | 1 request at 128K: 659 µs at 4% MFMA busy; 1 request at 1M: 5209 µs |
| vllm#65188 commit `3c5e29ffcd` (Rony Leppänen) | Triton MFMA tile scorer for prefill chunks of at least 768 rows, width trimmed to `max_seq_len`; served, real weights | median TTFT 28.8 to 30.0% lower at 128K (c1, c4) and 11.2 to 12.3% lower at 32K (c1, c4, c16); median ITL unchanged |

**Not measured yet:** current #5996 head against the vllm#65188 path (trimmed width and Triton MFMA scorer); the split of indexer time between scoring, top-k and expand at 128K to 1M prefill; any 1M prefill number on current heads; the indexer's share of TTFT after vllm#65188's scorer.

## First step: matched baseline and breakdown
One harness on MI355X with real vLLM geometry: compressed page 392, page table padded to `max_model_len`, strided cache view spanning more than 4 GiB, vLLM's row chunking included, eager timing for prefill and mixed steps, interleaved repeats, cold caches.

1. Arms: the vLLM AMD path at the vllm#65188 head; aiter#5996 head as wired by vllm#59437; vllm#62589 if still open.
2. Cases: one prompt of 8K, 32K, 128K and 256K (and 1M where `max_model_len` allows), processed as 16384-token chunks; two and four prompts in one step; a prompt chunk with 3, 16 and 64 decode rows; a prompt chunk with MTP decode rows (4 per request).
3. Report per QSA layer per step: scoring, top-k and expand time, chunk count, and score-buffer bytes written and read. Time sparse GQA alongside so the indexer share is known.
4. Decode, secondary: 1 to 64 rows and MTP rows at 8K to 256K under graph replay with padded tables, with the same split. No gate.

Stop or shrink the fused work if the score buffer and top-k are a small share of indexer time at 128K and beyond; the lever is then scorer efficiency, not fusion.

## Suggested approach (optional)
The goal is to stop writing `[M, columns]` fp32 scores and running a separate top-k. Two credible options; the first step should pick between them.

1. **Fused score with split-local top-k, then a split-aware merge** (the aiter#5525 structure). Keep #5996's multi-row tiles, give each workgroup a column span, keep each row's local top-512 over that span, fill the merge's first histogram while scoring, then merge per row. Prefill fills the GPU with row tiles, so few column splits are needed and each split is much wider than 512. At 128K (32,768 columns) four splits store 16 KiB of candidates per row instead of 128 KiB of fp32 scores; at 1M eight splits store 32 KiB instead of 1 MiB. What aiter#5525 does not have is per-row candidate state for 16 rows in one workgroup: 16 rows of 512 eight-byte candidates is 64 KiB, which fits gfx950 LDS but not gfx942's alongside the key tiles.
2. **Score twice, never store scores.** Scoring is cheap here (4 heads, BF16 MFMA; the #5996 scorer ran at about 4% MFMA busy). Pass 1 scores and builds each row's histogram of order-preserving 16-bit keys; pass 2 rescores and emits blocks above each row's threshold, collecting the boundary bin for an exact final pick. It reads the keys twice but drops the score-buffer write and read and the separate top-k passes. Histogram state needs sizing: 2048 int32 bins is 8 KiB per row, 128 MiB for 16384 rows, so use a coarser first level or chunk rows.

Either way: allow row tiles to span requests as #5996 does; take all blocks without scoring when `n_i <= 512`; write expanded token ids directly (fold expand); choose the variant per step on the host from `max_seq_len` and row counts. Decode rows of a mixed step may share the launch or keep their own. If neither option wins, a cheaper fallback is to keep the buffer but store 16-bit keys instead of fp32, halving its traffic and doubling rows per chunk, with an exact rescore at the boundary.

## Acceptance
1. Kernel: indexer time per QSA layer per step (scoring, top-k and expand, including chunking) beats the faster of the vllm#65188 path and aiter#5996 on prefill and mixed steps at 32K, 128K and 256K, for single and multi-prompt steps, with and without decode rows; within noise at 8K and below.
2. Correctness: the set gate above on every measured case, including padded tables, missing pages, wide cache views and mixed steps.
3. Serving: Qwen3.8-Flash-Next TP2 on MI355X through the vLLM opt-in path. Median TTFT at 32K and 128K inputs (c1, c4, c16) is lower than at the vllm#65188 head with its prefill scorer. Median ITL is not worse, and GSM8K and GPQA Diamond are unchanged.
4. Decode: a table, without a gate, of scoring, top-k and merge time at 1 to 64 rows and 8K to 256K for the fused design against the current paths, as input for the specialized decode kernels.
5. Closing with a profile only is valid if the first step shows the score buffer and top-k are a small share of long-context prefill indexer time.

## Open questions
* Should a mixed step's decode rows share the prefill launch, or keep a separate small launch?
* A served run of an earlier #5996 head (`4bbf029a9`, 2026-10-07) showed TTFT increases at higher concurrency, where steps are mixed: 128K c16 5853 to 6337 ms, 60K c16 2359 to 2422 ms, 8K c32 216 to 305 ms. They were not reproduced later in a different environment. Recheck them in the matched serving run.
* On gfx942, option 1's per-row candidate state does not fit in LDS next to the key tiles; option 2 or fewer rows per tile may be needed there.

## Coordination
* Land aiter#5996 and #6083 first, or build on their head.
* aiter#5525, #5677 and #5996 all edit `aiter/ops/flydsl/topk/topk_per_row.py`; agree a merge order with their owners. Johannes Graner now carries #5525.
* Related: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047) (base work), [SILOTIGER-1136](https://amd.atlassian.net/browse/SILOTIGER-1136) (the same question for GLM DSA prefill), [SILOTIGER-699](https://amd.atlassian.net/browse/SILOTIGER-699) and [SILOTIGER-1414](https://amd.atlassian.net/browse/SILOTIGER-1414) (decode top-k).

# [SILOTIGER-1047] [Qwen4-preview QSA] FlyDSL QSA indexer scorer + sparse GQA

[View in Jira](https://amd.atlassian.net/browse/SILOTIGER-1047) · Created 08/Sep/26 · Updated 15/Sep/26

| Field | Value | Field | Value |
|---|---|---|---|
| **Status** | Opened | **Type** | Story |
| **Project** | [Silo Tiger](https://amd.atlassian.net/secure/BrowseProject.jspa?id=13053) | **Priority** | Undefined |
| **Components** | [FlyDSL](https://amd.atlassian.net/issues/?jql=project%3D13053%20AND%20%22component%22%3D31844%20ORDER%20BY%20priority%20ASC), [Kernels](https://amd.atlassian.net/issues/?jql=project%3D13053%20AND%20%22component%22%3D25143%20ORDER%20BY%20priority%20ASC) | **Severity** | Medium |
| **Reporter** | Remes, Sami | **Assignee** | Aario, Sami |
| **Resolution** | Unresolved | **Votes** | 0 |
| **Affects versions** | None | **Fix versions** | None |
| **Labels** | None | **Sprint** | FlyDSL Sprint 5 |
| **Parent / Epic** | [SILOTIGER-1040 — Qwen4-preview (Qwen3.8-Flash-Next) kernels](https://amd.atlassian.net/browse/SILOTIGER-1040) | **Team** | — |
| **Original estimate** | Not specified | **Remaining estimate** | Not specified |
| **Time spent** | Not specified | | |

## TL;DR

Ship **FlyDSL** kernels for the Qwen Sparse Attention (QSA) block in Qwen3.8-Flash-Next / Qwen4-preview (`qwen4_exp`): (1) paged indexer **scorer** plus fused top-k, (2) **sparse GQA** attend on the selected tokens.

**Primary bar is whatever AMD serving actually launches today**, not the unmerged AITER PR. That is vLLM-vendored Triton plus HIP top-k in `qwen4_exp/amd/ops/qsa.py` (vLLM PR 53896). Beat that end-to-end on one QSA layer (`indexer + select + attend`) before claiming a win.

**Secondary bar** is the unmerged AITER Triton/Gluon stack in [ROCm/aiter#4882](https://github.com/ROCm/aiter/pull/4882): beat its portable Triton path, and beat its gfx950 Gluon path **on the shapes Gluon actually dispatches**. #4882 is not on `main` and is not the vLLM default; treat it as a competitor, not as production.

QSA is **not** every layer: hybrid is `GGGQ` x 12, so **12 of 48** layers (plus MTP, which can reuse indices).

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040). Sibling GR work is [SILOTIGER-1042](https://amd.atlassian.net/browse/SILOTIGER-1042) / [SILOTIGER-1041](https://amd.atlassian.net/browse/SILOTIGER-1041); do not mix GR into this ticket.

## Why a new kernel

At long context a QSA layer is two jobs with different scaling:

1. **Indexer** scores every **complete 4-token block** (`O(L/4)` keys), top-512, expand plus tail to at most 2051 token ids. Cost grows with `L`.
2. **Sparse GQA** attends those ids on uncompressed K/V. Cost is almost independent of `L` once the budget is full (`K = 2048` tokens, `r = 4` blocks).

The architecture report's 7.6x prefill / 4.9x decode at 1M is vs **dense** GQA and includes both. AMD serving today is vLLM-vendored Triton (scalar MQA loop, sparse GQA `num_stages=1` even on gfx950). #4882 is AITER Triton plus optional gfx950 Gluon; it is **not on `main`**, not wired as the vLLM default, and Gluon sparse GQA does **not** match the released Flash-Next GQA shape.

Do not retarget GLM/DeepSeek DSA: that scorer is `H=32` FP8 with per-head `w_h` over **tokens**. QSA is `H=4` BF16, no `w_h`, over **mean-pooled blocks**. Do not retarget sparse MLA prefill or SWA.

## Two shape families (both gated)

### A — Flash-Next production (must win)

Checkpoint `Qwen/Qwen3.8-Flash-Next` / FP8 twin. Config: `full_attention_interval=4`.

Indexer:

- `indexer_n_heads = 4`, `indexer_kv_heads = 1`, `indexer_head_dim = 128`
- `indexer_compress_ratio = 4`, `indexer_budget = 2048` so `K_B = 512` blocks
- Score: `I_ib = sum_h ReLU(dot(q[h], k_bar[b]))` for complete blocks only (`p_b + r - 1 <= i`). No learned per-head weight. Serving may apply `1/sqrt(128)`; that cannot change top-k argmax.
- Then `TopK` of 512, expand each block to 4 tokens, union the incomplete tail (0..3 tokens). Output width at most 2051.

Sparse GQA:

- 24 Q heads, 2 KV heads, so group size 12, `head_dim = 256`, partial RoPE 64, sigmoid output gate
- Attend uncompressed paged K/V at the expanded token indices
- BF16 activations (owner requires BF16 main QSA cache, not FP8 KV)

### B — #4882 Gluon-validated shapes (parity / competitor)

Keep these so FlyDSL is measured on the same points Gluon was tuned for, not only the checkpoint.

Indexer (Gluon auto-dispatch in #4882): gfx950, Triton `>= 3.6`, `H` is 4 or 8, `D = 128`. Their published bench: `rows=32`, `heads=4`, `head_dim=128`, `page_size=8`, `pages=512`. Reported p50: Triton 0.0184 ms, Gluon 0.0176 ms (**1.047x**).

Sparse GQA (Gluon auto-dispatch): `head_dim = 128`, GQA **group size 5**, `selection_width = 2051`. Their published bench: `num_tokens=16`, `num_query_heads=10`, `head_dim=128`, width 2051. Ordered: Triton 0.116 ms, Gluon 0.092 ms (**1.25x**). Randomized indices similar.

**Released Flash-Next GQA is group 12 and D=256, so #4882 Gluon will not auto-dispatch there.** Family A vs #4882 is Triton-vs-FlyDSL. Family B vs #4882 is Triton-and-Gluon-vs-FlyDSL.

## Kernel structure (FlyDSL)

Two kernels, one op surface under `aiter/ops/flydsl/`.

**K1 — scorer plus fused top-k.** Stream paged compressed index-K tiles, compute the ReLU-sum score, keep a **local** top-512 (or local top-k for family B), merge to a global 512 (or k) without writing `[rows, n_blocks]` FP32 scores. Same "score-plus-top-k" idea as DSA fused indexer work; different ABI (`H`, no `w_h`, block keys, `k=512`). Expand-plus-tail can live in K1's epilogue or K2's prologue.

Existing FlyDSL `fp8_mqa_logits` is the wrong kernel: it needs `H % 16 == 0`, is dense-only, and uses weighted ReLU. Pad-to-16 is a prototype only.

**K2 — sparse GQA.** 24x2 (family A) or 10x2 / group 5 (family B). Selected positions are 512 runs of 4 plus a short tail on A, or a 2051-wide index list on B. Split-K as needed. Compile gfx942 and gfx950 separately if LDS/VGPR models differ; gfx950 should use the extra LDS (#4882 and vLLM both left `num_stages=1` on the vLLM AMD path).

Prefetch K/V for the selected runs. Do not union decode GEMV and prefill MFMA in one instantiation if that costs occupancy.

## Baselines (live path first)

Report each named backend separately. Do not hide a loss to vLLM behind a win vs #4882, or the reverse.

### Live AMD serving (must beat)

This is the active path in `qwen4_exp/amd/` today:

| Step | What runs |
|---|---|
| Indexer Q/K GEMM | vLLM unquant linear: `wvSplitK` (tokens 1–5), hole at 6–9, else `F.linear` / hipBLASLt; gfx950 may add `wvSplitKrc` / AITER tgemm if those linears are on |
| `RMSNorm + partial MRoPE` | unfused `GemmaRMSNorm + triton_mrope` (no `qsa_pre_indexer.py` on AMD) |
| `Compress r=4 + paged store` | Triton `qsa_compress_groups_with_ratio` / `qsa_store_cache_rows` |
| Paged MQA scores | Triton `_qsa_mqa_paged_kernel` — **scalar per-head loop, no `tl.dot`** |
| Top-k (512 blocks) | HIP `top_k_per_row_decode` in `csrc/libtorch_stable/sampler.cu` |
| `Expand + tail` | Triton `_expand_qsa_indices_kernel` |
| Sparse GQA | Triton `_qsa_sparse_paged_gqa_splitk_kernel`, **`num_stages=1`** (comment assumes 64 KiB LDS on gfx942 and gfx950) |

Files: `vllm/models/qwen4_exp/amd/ops/qsa.py` plus the Linear / HIP top-k call sites (vLLM PR 53896). Same Triton on gfx942 and gfx950 except skinny GEMM extras on gfx950.

End-to-end gate is this **whole chain** (launches, bytes, and fused K1/K2), not a single kernel vs its Triton twin in isolation.

### Not live on AMD (reference / competitor only)

- AITER #4882 Triton — `aiter/ops/triton/_triton_kernels/attention/qsa_paged_mqa_logits.py` and `qsa_sparse_paged_gqa.py`. Opt-in AITER; SGLang can use it eager-only. Competitor on family A GQA, where Gluon does not dispatch.
- AITER #4882 Gluon — `aiter/ops/triton/_gluon_kernels/gfx950/attention/`. gfx950, Triton `>= 3.6`. Indexer: `H` is 4 or 8, `D=128`. Sparse GQA: `D=128`, group size 5, width 2051. Forced `gluon` on a miss must error; `auto` falls back to Triton. If #4882 merges, retarget to `main`. Until then pin a PR head (runtime-tested parent `2462d5b64`; later heads may be docs-only).
- vLLM NVIDIA QSA Triton — `qwen4_exp/nvidia/ops/qsa.py`: tensor-core MQA (`tl.dot`), sparse GQA `num_stages=2`, fused `qsa_pre_indexer`, CUDA `persistent_topk` / `cooperative_topk`. This is not the AMD acceptance bar. Keep it as an optional column in the result table so a FlyDSL win vs AMD Triton is not confused with catching NVIDIA.

AITER fused MoE, tgemm, and vision FA may be on in the same process; they are **not** QSA baselines. Do not compare against FlashInfer TRT-LLM QSA (NVIDIA recipe even passes `--no-enable-flashinfer-autotune`).

## Interface (family A)

Indexer in: BF16 `q` `[M, 4, 128]`, paged compressed `k` (one KV head, D=128), block table, per-row causal complete-block bound, `eps` unused. Optional scale `1/sqrt(128)`.

Indexer out: `block_ids` `[M, 512]`, then token `indices` `[M, <=2051]` after `expand+tail`.

GQA in: BF16 `q` `[M, 24, 256]`, paged `k`/`v` `[..., 2, 256]`, `indices`, softmax scale, sigmoid gate weights if fused into the epilogue.

GQA out: BF16 `o` `[M, 24, 256]` (pre-`o_proj`).

`M` is flattened tokens. Decode `1..8` and prefill 512 / 2048 / 8192 **tokens** — prefill moves `M` and `L` together — plus at least one long-context length (32k or 128k) so indexer scaling is visible. gfx942 and gfx950.

The harness sweeps `M` and `L` as independent axes and takes the cross-product, so it also emits decode rows at short `L`, which nothing above asks for. **K2's bar is the budget-saturated points: decode `M∈{1,8}` at `L=32768`, prefill `M=512` at `L=8192`.** Below `L=2048` the 2051-wide selection is mostly `-1` padding — a quarter live at `L=512` decode, an eighth at `M=512` prefill — so the kernel spends most of its tiles on masked columns that all clamp to page 0 and stay cached. Those rows measure the masked path, not the gather, and they disagree with the saturated verdict: ±5% at decode, and inverted at prefill (K2/AMD is 0.82× at `L=512` but 1.42× at `L=8192`). Keep them as fast smoke rows; the bench reports `valid%` so the dilution is visible in the table. `M=512 L=512` is a genuine 512-token prefill and is on the list above on its own merits.

## Correctness

Independent fp32 oracle: block-causal ReLU-sum scores, exact top-512 (tie-break documented), `expand+tail`, then standard GQA on those positions. Cross-check vs vLLM Triton `qsa.py` and vs #4882 on shared shapes. Authoritative math: Qwen3.8-Next tech report §2.1 / QSA, [tech_report.pdf](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf). Config: [config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json).

Fusing top-k may change fp32 score order vs materializing full logits; gate K1 on the oracle with a documented tolerance, and require **set equality** of selected blocks (or a fixed tie policy).

## Phases

1. Harness: pin **vLLM AMD live path**, #4882 Triton, #4882 Gluon (where it dispatches), and a fp32 oracle. Report family A and family B separately. rocprof one real QSA layer (indexer through GQA) at short and long `L`, including HIP graph replay at decode.
2. FlyDSL K1 on family A (`H=4`) and family B (`H` is 4 or 8). Gate: beat live vLLM AMD (`MQA Triton + HIP top-k`) and beat #4882 Triton; beat Gluon on gfx950 where Gluon dispatches. No full score matrix.
3. FlyDSL K2 on family A (`D=256`, group 12) vs live vLLM AMD sparse GQA and #4882 Triton. Then family B (`D=128`, group 5) vs #4882 Triton **and** Gluon.
4. Wire `aiter/ops/flydsl/` plus vLLM `qwen4_exp` opt-in, same three-way backend idea as #4882 (`auto` / FlyDSL / Triton).
5. Optional: fuse `expand+tail` into K2; fuse `qsa_pre_indexer` (`Gemma RMSNorm + partial MRoPE + compress`) only after K1/K2 beat the bar.

Warm up by duration. Interleave paired rounds. Do not claim a GQA win from the group-5 D=128 Gluon bench, and do not claim an indexer win from DSA `H=32` FP8 numbers.

## Related code

- Baseline serving: `vllm/models/qwen4_exp/amd/ops/qsa.py`, NVIDIA twin `qwen4_exp/nvidia/ops/qsa.py` (tensor-core MQA, `num_stages=2`; not the AMD bar).
- Competitor: [ROCm/aiter#4882](https://github.com/ROCm/aiter/pull/4882) (`aiter/ops/triton/attention/qsa.py`).
- Wrong ops: FlyDSL `fp8_mqa_logits` (`H % 16 == 0`), MLA sparse decode, SWA, AITER mHC, GR tickets above.
- Neighbors: MoE tune CSVs [ROCm/aiter#5213](https://github.com/ROCm/aiter/pull/5213); GDN is a different 36-layer problem.

## Open questions

- Where indexer vs GQA dominates on AMD at 8k / 32k / 128k / 1M; that sets which kernel to land first after the harness.
- Whether FlyDSL K1 should emit block ids or already-expanded token ids.
- Packed vs padded `M` (varlen) from a real vLLM prefill trace.
- MTP IndexShare (`indexer.skip_topk`): GQA-only launch, no scorer.
- #4882 merge timing: keep competing even if it lands; do not wait on it to start the harness.

---

Generated at Tue Sep 15 11:07:39 UTC 2026 by Aario, Sami using Jira 1001.0.0-SNAPSHOT#100294-rev:b0380760584d59a844a637560bce3ca4d5aec84d.

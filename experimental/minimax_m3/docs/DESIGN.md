# MiniMax-M3 decode monokernel in vLLM: design

Goal: run MiniMax-M3 decode in vLLM's own AMD model (`vllm/models/minimax_m3/amd/model.py`) on a
resident persistent kernel, with two interchangeable backends behind one switch:

- `flydsl`: ATOM's FlyDSL M3 mono (`ROCm/ATOM` `atom/models/minimax_m3/mono`, HEAD 67c54da), ported.
- `gluon`: a new Gluon (Triton 3.8, gfx950) implementation of the same contract.

`VLLM_M3_MONO=1 VLLM_M3_MONO_BACKEND=flydsl|gluon`; off = vLLM's original path. Then vLLM
1k/1k concurrency-1 serving for off / flydsl / gluon.

## Scope

- Mono covers the 57 sparse MoE layers (3..59). One launch per layer per rank: the layer's pre-attention
  (K1) fused in front of attention .. FFN (K4), both TP all-reduces in-kernel.
- Dense layers 0..2 stay on vLLM's original path. ATOM runs them as three launches around its own
  attention kernel on an ATOM-specific cache layout; vLLM's dense layers use an unshuffled TRITON_ATTN
  cache, so porting them buys ~3 of 60 layers for a large contract change. Revisit later.
- Decode of 1..16 token rows (graph pad rows included), TP4, 256 CUs, fp8 KV, block 128. Anything else
  falls back to the original path for the whole step (the decision is TP-uniform).

## Layer hand-off (matches vLLM's own)

vLLM layer i returns `(hidden, residual)`: hidden = FFN output already all-reduced, residual = running
sum before the norm; layer i+1 runs `gemma_fused_add_rmsnorm(hidden, residual)`. ATOM's mono uses the same
pair as `(ar, h_mid)`. So mono takes layer 2's `(hidden, residual)` and returns layer 59's, and vLLM's final
`norm(hidden, residual)` + lm_head run unchanged. Ping-pong buffers by layer parity, as in ATOM.

## Contract differences and how the port resolves them

| item | vLLM (what serving writes / reads) | ATOM mono expects | port |
|---|---|---|---|
| qkv (+index q/k) and o_proj weights | BF16 [2560,6144], [6144,2048] row-major | ptpc FP8 (row scale amax/448) preshuffled (16,16) | quantize at mono setup: per-row FP8 + `shuffle_weight(16,16)`; o_proj row scale over the full K (all-reduce MAX of row amax across ranks, as ATOM gathers the row). Extra ~1.6 GB/rank. Numerics follow ATOM (FP8 attention GEMVs), not vLLM's BF16 GEMMs. |
| K/V cache | fp8, **static unit scale**, page-16 views over [nb,2,128,128]: K pages 0..7, V pages 8..15 of each 128-token block; page16 slot = slot + (slot//128)*128 | separate K / V page-16 tensors, **per-token** fp32 scales | keep vLLM's cache contract (prefill and the original path share the cache): insert with unit scale; ATOM's sparse K offset formula holds with page16 slots; V base = K base + 8 pages. Attention reads unit scales. |
| index cache | **BF16** [nb,128,128] (Triton indexer; AITER fp8 indexer needs `indexer_kv_dtype` fp8) | FP8 E4M3 unit scale | keep BF16 (no serving-config change, baseline unchanged): insert bf16, scorer reads bf16 K (vLLM's Triton scorer is bf16 K x bf16 q). |
| router | FP32 gate GEMM, fp32 logits, `biased_grouped_topk` (sigmoid, bias for selection, renorm x2) | BF16 gate view, bf16-rounded logits | first port: ATOM numerics (bf16 copy of the gate); then match vLLM (fp32 dot) if routing diverges. |
| experts | aiter-shuffled MXFP4 w13 [129,1536,3072] (gate rows then up rows), w2 [129,6144,384], e8m0 shuffled scales; shared expert fused as id 128, weight 1.0 | same family (`moe_shuffle_weight` gate_up, `moe_shuffle_scale`), 129 experts, shared weight from the topk meta | verify byte equality on a loaded model (test); read shared weight from vLLM's `aiter_topK_meta_data`. |
| sparse metadata | `MiniMaxM3SparseMetadata`: `page16_slot_mapping`, `.decode.block_table` (128-token), `.decode.seq_lens`, `decode_query_len`; no `batch_id_per_q_token` | `slot_mapping`, `block_table`, `seq_lens`, `batch_id_per_q_token` (pad rows -1), `max_query_len` | pass page16 slots; derive row liveness on device (seq_len > 0 / slot >= 0); expand spec-verify rows like ATOM's `token_rows`. |
| indexer reuse | layers with sparse ordinal % 4 == 0 select (3, 7, .., 59); others reuse | `index_topk` per layer; first sparse layer must select | same rule from `self_attn.skip_index_topk`; selection kept in mono scratch across the step. |
| all-reduce / peer memory | vLLM custom AR | ATOM `UncachedIpcHeap` peer buffer + epoch mailboxes | own symmetric peer buffer (as vLLM's DSv4.1 mono `PeerBuffer`); runner created eagerly before graph capture. |
| engine coupling | vLLM forward context, TP group | ATOM forward context, envs, consensus, topk meta | thin vLLM shims for `get_forward_context`, envs, TP group/consensus, topk meta; vendor `atom/mono/{device,plan,runtime}` (~83 KB) and `atom/models/minimax_m3/mono` under `vllm/models/minimax_m3/amd/mono`. |

## Integration (same pattern as vLLM's DSv4.1 mono)

- `MiniMaxM3Model.forward` (decoder loop): after layer 2, if mono accepts the step, run layers 3..59 in
  the runner and skip their modules. Weight views and FP8 copies built once (lazily before the first
  capture), runner singleton per process, declines eagerly-shaped/unsupported steps.
- Decode runs as FULL_DECODE_ONLY graphs (sizes 1, 2 with max_num_seqs 2): the runner reads metadata
  from persistent buffers and must be capturable.

## Gluon backend

Same contract and stage graph as the FlyDSL K1+K4 (index, select, split attention, combine, o_proj +
AR, router, MoE narrow/wide, down + AR), 256 x 512 persistent grid, tagged-pair mailboxes and epoch
fences. Carry over the DSv4.1 lessons: `uniform()` (readfirstlane) for every buffer base derived from
loaded/LDS data (else waterfall loops), stage B scales in LDS (a global load in a prefetched K loop drains
vmcnt), `pin()` to keep software prefetch, issue weight loads before input waits, put each post-seam stage
on CTAs that are free when its input lands, multi-element polls via one coherent peek. Known Gluon limit:
a 2-D dot shares B across waves, so split-K over waves re-reads B (prefer row-parallel tiles with full K
per wave where task counts allow).

## Validation ladder

1. Per-kernel numerics vs the original vLLM path on a loaded model (one layer, TP4, widths 1/2/4/8/16).
2. All 57 layers: logits vs original path; greedy text identical or near on a prompt set.
3. Graph capture + replay; serving 1k/1k c=1 for off / flydsl / gluon (one benchmark at a time, GPUs 4-7).

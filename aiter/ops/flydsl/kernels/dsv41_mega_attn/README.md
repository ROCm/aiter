# DeepSeek-V4.1-Flash attention mega kernel (gfx950, FlyDSL)

Resident, task-table kernels that run one DeepSeek-V4.1 attention layer of a
decode step in two launches instead of the ~12 kernels vLLM's ROCm path runs
today. The design is adapted from ATOM's V4.1 mono decode
(`atom/models/deepseek_v41/mono`, ROCm/ATOM 0873517), which itself follows the
FlyDSL GLM-5 / Kimi-K3 monokernel lineage: one CTA per CU, all CTAs resident,
each stage a table of tasks placed round the CTAs, hand-offs through tagged
mailboxes, and every consumer's weights in flight before it waits on its input.

## Scope (phase 1)

| | |
|---|---|
| Layers | every layer without its own compressor or indexer: 32 of 40. Ratio 1: 21-23, 25-27, 29-31, 33-35, 37-39 (top-k over layer 20's cache, 128 entries a page); ratio 2: 3-7, 9-13, 15-19 (top-k over layer 2 / 8 / 14's cache, 64 entries a page); ratio 0: 0-1 (window only, plain RoPE). Layers 2, 8, 14, 20 (compressor + indexer) and 24, 28, 32, 36 (indexer) keep the original path, since their attention needs the same layer's fresh top-k |
| Batches | decode only, M <= 48 rows (concurrency 1-8 x 1-6 tokens a request) |
| TP | 2 and 4 (32 / 16 local heads, 4 / 2 local wo_a groups) |
| Device | MI355X (gfx950), 256 CUs |
| Input | `hidden` [M, 5120] bf16: the attention seam's normed row (vLLM's mHC kernel already applied attn_norm) |
| Output | `out` [M, 5120] bf16: this rank's wo_b partial; vLLM's all-reduce follows as today |
| Side effect | the step's KV rows written into the layer's sliding-window cache |

Anything else (prefill or mixed batches, M > 48, profiling runs) takes the
original path.

## What it replaces (vLLM ROCm path, TP2, M = 6, per layer)

| op | kernel(s) | us |
|---|---|---:|
| fused_wqa_wkv | `mxfp8_quantize` + `rocm_mxfp8_block32_gemm` | 10.6 |
| q/kv RMSNorm | `FusedQKVRMSNorm` | 3.2 |
| wq_b | `mxfp8_quantize` + `rocm_mxfp8_block32_gemm` | 9.1 |
| Q RoPE, KV RoPE + quant + SWA insert | `fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel` | 5.8 |
| sparse attention | aiter `_sparse_mla` + `_sparse_mla_reduce` | 19.4 |
| inverse RoPE + MXFP8 | `_inverse_rope_mxfp8_quant_kernel` | 4.6 |
| wo_a | `_mxfp8_wo_a_bmm_kernel` | 6.3 |
| wo_b | `mxfp8_quantize` + `rocm_mxfp8_block32_gemm` | 12.3 |
| **total** | **12 kernels** | **~71** |

plus, under breakable CUDA graphs, the host gap of the eager attention segment.

## Semantics (what the kernels must reproduce)

1. `x8 = mxfp8(hidden)`; `qr_kv = x8 @ wqkv^T` (wqkv = [wq_a; wkv], [1792, 5120]
   e4m3, 32x32 E8M0 block scales) -> bf16.
2. `q = rmsnorm(qr_kv[:, :1280], q_norm)`, `kv = rmsnorm(qr_kv[:, 1280:], kv_norm)`
   (fp32, eps = 1e-20) -> bf16.
3. `q8 = mxfp8(q)`; `Q = q8 @ wq_b^T` -> bf16 [M, H, 512]; GPT-J RoPE of dims
   448..511 (fp32, the layer's fp32 `cos_sin_cache`) -> bf16.
4. KV: GPT-J RoPE of dims 448..511 -> bf16; dims 0..447 as e4m3 with one UE8M0
   per 64 (amax floored at 1e-4, ceil exponent), dims 448..511 bf16; stored as
   the V4 `fp8_ds_mla` record (584 B: 576 B data rows, then 8 B of scales per
   token after the block's rows) at `slot_mapping[t]` when >= 0.
5. Attention, per token and head: keys = the sliding window
   (`decode_swa_indices[t, :decode_swa_lens[t]]`, slots of the layer's SWA
   cache) and the top-k compressed rows (`topk_indices_buffer[t, :]` >= 0,
   mapped through the kv-source cache's block table: 128 entries a page at
   ratio 1, 64 at ratio 2, no compressed rows at ratio 0); K / V =
   the dequantized 512-wide row; bf16 dots, fp32 accumulation,
   `softmax(scale * q.k)` in base 2 with the attention sink folded in:
   `O = sum(p v) / (sum(p) + exp(sink - m))`, bf16.
6. Inverse GPT-J RoPE of dims 448..511 (fp32) -> `mxfp8` -> wo_a's input.
7. wo_a: `Z[:, g] = A[:, 4096 g : 4096 (g + 1)] @ wo_a[1024 g : 1024 (g + 1)]^T`
   -> bf16 [M, G * 1024].
8. `out = mxfp8(Z) @ wo_b^T` -> bf16 [M, 5120].

`mxfp8(x)`: one E8M0 per 32 values, `code = clamp(ceil(log2(amax / 448)) + 127,
0, 254)`, values `e4m3(x * 2^(127 - code))` (vLLM's ROCm activation quant).
Weights stay in vLLM's loaded layout (row-major e4m3 [N, K], [N/32, K/32] E8M0),
so the original path and the kernels share one copy.

## Execution model

* Grid = 256 CTAs (one a CU, all resident), 512 threads (8 waves) a CTA,
  LDS a per-stage union.
* A stage is a task table; CTA `b` runs tasks `b, b + 256, ...` of each stage,
  stages in order. A consumer only waits on earlier stages, whose tasks every
  CTA finishes before reaching it: no deadlock while the grid is resident.
* Hand-off: tagged-pair mailbox (value + tag in one 8 B store, device scope),
  polled until every tag matches; the tag is `epoch * 64 + layer + 1`, where
  `epoch` is a device counter the model bumps once a step, so no mailbox is
  cleared between steps and the launch stays graph-capturable (every argument
  is a fixed device pointer).
* Weights stream nontemporal; a GEMV task issues its weight loads before it
  polls its activations, so weight latency hides behind the producer stage.

### K1 `front` (hidden -> Q, KV insert)

| stage | tasks | per task |
|---|---|---|
| quant | M | a row -> X8 / X8S |
| wqkv_a | 112 (16 rows) | 16 x 5120 e4m3 GEMV over every row -> QKV (bf16) |
| qkv | M | q / kv RMSNorm; q -> QX8 / QX8S; kv RoPE + record quant -> SWA cache |
| wq_b | 1024 / 512 (16 rows) | 16 x 1280 e4m3 GEMV -> RoPE -> Q (global, bf16) |

### K2 `back` (attention -> wo_b partial)

| stage | tasks | per task |
|---|---|---|
| score | M x H/16 x splits | 16 keys a split a wave: gather + dequant, QK (bf16 MFMA), m / l / bf16 p -> AM / AL / AP |
| pv + irq | M x H/16 x column groups | PV over the splits, alpha / sink combine, inverse RoPE, mxfp8 -> XO / XOS |
| wo_a | 128 / 64 (32 rows) | 32 x 4096 e4m3 GEMV (its group's rows) -> bf16 -> mxfp8 (one group) -> X8B / X8BS |
| wo_b | 320 (16 rows) | 16 x (G * 1024) e4m3 GEMV -> `out` (bf16) |

The launch boundary between K1 and K2 publishes the step's KV rows and Q to the
attention without device-scope traffic; merging the two is a later step.

## vLLM integration

`aiter.ops.flydsl.dsv41_mega_attn.DSV41MegaAttention` owns the scratch,
epoch and compiled launchers for one TP rank. The vLLM ROCm DSv4.1 attention
layer calls it from its eager attention segment (and, in FULL uniform-decode
graphs, inside the capture) when the batch is decode-only with M <= 48;
everything else takes the original path. The device inputs are the ones the
original decode reads: positions, the SWA metadata's persistent
`slot_mapping` / `decode_swa_indices` / `decode_swa_lens` /
`token_to_req_indices`, the shared `topk_indices_buffer`, and the compressed
cache's block table.

## Validation

* Kernel level: real layer weights (TP shards of the checkpoint), caches filled
  through vLLM's own insert op, the reference being the exact op chain vLLM
  runs. Compared: `out` and the written SWA records (max / mean abs error,
  cosine), every M in 1..48 and TP 2 / 4; timed under HIP graphs against the
  reference chain.
* End to end: AgentX replay with the SemiAnalysisAI/InferenceX#3690 recipe,
  TP2 c1 / c2 / c4 / c8, baseline and mega side by side on all eight GPUs.

## Later

* M up to 1152: token-tiled GEMMs (MFMA tiles over rows as well as columns).
* One launch a layer (in-launch KV hand-off).
* The TP all-reduce in the kernel (peer push, as ATOM's K2a).
* The indexer layers (a front / back split around the indexer).

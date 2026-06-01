# Gluon Kernel Status

All kernels in this directory are written in Gluon, a GPU programming language at the same level as Triton but with more explicit control over layouts, async copy, and MFMA intrinsics.
Some features (e.g., scheduling hints like `sched_barrier`) require the [AMD Gluon Extension](https://github.com/ROCm/triton/tree/gluon_ext).

## Quick Reference

<small>
<table>
<tr>
  <th rowspan="2">Kernel</th><th rowspan="2">Op</th><th rowspan="2">Arch</th><th rowspan="2">Constraints</th>
  <th rowspan="2">Typical Test</th>
  <th colspan="3">Perf of the Typical Test</th>
</tr>
<tr>
  <th>Gluon</th><th>ASM</th><th>CK</th>
</tr>
<tr>
  <td><code>gemm_a8w8</code></td><td>GEMM</td><td>CDNA4</td>
  <td nowrap>A: int8/fp8 (e4m3/e5m2)<br>B: int8/fp8 (e4m3/e5m2)<br>Out: bf16/fp16<br>Tunable BLOCK_M/N/K</td>
  <td>python op_tests/triton_tests/<br>gemm/basic/test_gemm_a8w8.py</td>
  <td>TBD</td><td>—</td><td>TBD</td>
</tr>
<tr>
  <td rowspan="3"><code>mla_decode_gluon</code></td><td rowspan="3">MLA<br>Decode</td><td rowspan="3">CDNA4</td>
  <td nowrap>(bh64)<br>Q: bf16, KV: bf16, Out: bf16<br>batch_size in {64, 128, 256}<br>nhead in {64, 128}<br>PAGE_SIZE=1<br>BLOCK_H=BLOCK_N=64</td>
  <td>python op_tests/test_mla.py \<br>-c 16384 -b 64 128 \<br>-n 64,1 128,1 \<br>-d bf16 -kvd bf16</td>
  <td>~563<br>TFLOPS</td><td>~477<br>TFLOPS</td><td>—</td>
</tr>
<tr>
  <td nowrap>(bh16bn128)<br>Q: bf16, KV: fp8, Out: bf16<br>batch_size = 1<br>nhead &le; 16<br>PAGE_SIZE=1<br>BLOCK_H=16, BLOCK_N=128</td>
  <td>python op_tests/test_mla.py \<br>-c 10000000 -b 1 -n 16,1 \<br>-d bf16 -kvd fp8</td>
  <td>~4.58<br>TB/s</td><td>—</td><td>—</td>
</tr>
<tr>
  <td nowrap>(bh16bn64)<br>Q: bf16, KV: bf16, Out: bf16<br>batch_size = 1<br>nhead &le; 16<br>PAGE_SIZE=1<br>BLOCK_H=16, BLOCK_N=64</td>
  <td>python op_tests/test_mla.py \<br>-c 10000000 -b 1 -n 16,1 \<br>-d bf16 -kvd bf16</td>
  <td>~5.33<br>TB/s</td><td>~0.69<br>TB/s</td><td>—</td>
</tr>
<tr>
  <td><code>pa_decode_gluon</code></td><td>Paged Attn<br>Decode</td><td>CDNA3<br>CDNA4</td>
  <td nowrap>Q: fp8/bf16/fp16<br>KV: fp8/bf16/fp16<br>Out: bf16 or match<br>query_len &le; 4<br>query_len &times; group_size &le; 64<br>ctx_partition = 256</td>
  <td>python op_tests/triton_tests/<br>test_pa_decode_gluon.py</td>
  <td>TBD</td><td>TBD</td><td>TBD</td>
</tr>
<tr>
  <td><code>sparse_attention_dsv4</code></td><td>Sparse MLA<br>Prefill+Decode</td><td>CDNA4</td>
  <td nowrap>Q/Out: bf16/fp16<br>head_dim=512 (NoPE 448 + RoPE 64)<br>Prefill KV: bf16 flat [N, D]<br>Decode KV: fp8 NoPE + bf16/fp16 RoPE</td>
  <td>python op_tests/<br>test_sparse_attention_dsv4.py \<br>--cfgs 4096,128,4096,512</td>
  <td>~380<br>TFLOPS</td><td>—</td><td>—</td>
</tr>
</table>
</small>

---

## GEMM Kernels

### `gemm_a8w8.py` — INT8/FP8 GEMM

**Functions:** `gemm_a8w8(x, w, x_scale, w_scale, bias=None, dtype=bf16, y=None, config=None)`, `gemm_a8w8_preshuffle(...)`

**Description:** C = A &times; B^T with per-tensor row/column scales and optional bias. The `preshuffle` variant expects weights in a pre-shuffled `[N*16, K//16]` layout for better memory access.

| Parameter | Details |
|-----------|---------|
| Arch | gfx950 (CDNA4) only |
| A dtype | int8, fp8_e4m3, fp8_e5m2 |
| B dtype | int8, fp8_e4m3, fp8_e5m2 |
| Output | bf16 or fp16 |
| Scales | per-row (A), per-column (B), float32 |
| Tunable | BLOCK_SIZE_M/N/K, GROUP_SIZE_M, NUM_XCDS, NUM_WARPS |
| Config | `$AITER_TRITON_CONFIGS_PATH/gemm/gluon/gfx950-GEMM-A8W8.json` |

---

## Attention Kernels

### `mla_decode_gluon.py` — MLA Decode

**Function:** `mla_decode_gluon(q_nope, q_pe, kv_c, o, page_table, seq_info, sm_scale, k_pe=None, kv_pe_offset=512, use_2d_view=True, kv_scale=1.0, min_kv_seq_len=1)`

**Description:** Multi-head Latent Attention (DeepSeek MLA) decode kernel with split-KV. Q is split into compressed latent (`q_nope`, dim=kv_lora_rank) and rope positional encoding (`q_pe`, dim=qk_rope_head_dim). KV cache is a flat `[N, 576]` buffer (`kv_c`). Uses 3-stage async copy pipeline with double-buffered page numbers and KV tiles.

The wrapper dispatches by `(nhead, kv_c.dtype)` to one of three compile-time regimes (single `@gluon.jit` kernel, REGIME constexpr gates layouts and grid mapping):

- **`bh64`** (`nhead in {64, 128}`): bf16 KV, BLOCK_H=64, BLOCK_N=64, multi-batch + XCD-aware grid. `NUM_KV_SPLITS` auto-picked &isin; {1, 2, 4} so the launch fills ~256 workgroups (one wave on MI350). When `NUM_KV_SPLITS == 1`, stage-1 writes the final attention output directly to `o` (no temp buffer, no reduce). When `NUM_KV_SPLITS > 1`, stage-1 writes per-split `(acc, lse)` to a temp buffer and stage-2 (`_mla_softmax_reducev_kernel`) reduces them into `o`.
- **`bh16bn128`** (`nhead &le; 16`, `batch_size == 1`, fp8 KV): BLOCK_H=16, BLOCK_N=128, 1-D grid `(NUM_KV_SPLITS=256,)`. Optional `kv_scale` dequant. Always splits + always runs stage-2 reduce. `NHEAD < BLOCK_H` masks OOB heads on Q load and O store (wasted MFMA lanes are free; this regime is memory-bound).
- **`bh16bn64`** (`nhead &le; 16`, `batch_size == 1`, bf16 KV): BLOCK_H=16, BLOCK_N=64, same 1-D grid and split-then-reduce as bh16bn128. Use when KV is kept in bf16 (no fp8 quant). Same `NHEAD < BLOCK_H` masking.

Modified from [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).

| Parameter | `bh64` regime | `bh16bn128` regime | `bh16bn64` regime |
|-----------|---------------|--------------------|--------------------|
| Arch | gfx950 (CDNA4) | gfx950 (CDNA4) | gfx950 (CDNA4) |
| Q dtype | bf16 | bf16 | bf16 |
| KV dtype | bf16 | fp8 | bf16 |
| Output | bf16 | bf16 | bf16 |
| batch_size | 64, 128, or 256 | 1 | 1 |
| nhead | 64 or 128 | &le; 16 (tested: 4, 8, 16) | &le; 16 (tested: 4, 8, 16) |
| Page size | 1 | 1 | 1 |
| BLOCK_H | 64 | 16 | 16 |
| BLOCK_N | 64 | 128 | 64 |
| MFMA | 16&times;16&times;32, warps=[4,1] | 16&times;16&times;32, warps=[1,4] | 16&times;16&times;32, warps=[1,4] |
| NUM_KV_SPLITS | auto &isin; {1, 2, 4} from (batch, nhead) | 256 (fixed) | 256 (fixed) |
| `kv_scale` | unused (pass 1.0) | dequant scale folded into `qk_scale` (applied before softmax for fp8 correctness) | unused (pass 1.0) |
| Seq constraint | `min_kv_seq_len > NUM_KV_SPLITS * (3 * BLOCK_N + NUM_KV_SPLITS)` (the `3` matches the kernel's `gl.assume(num_iter > 3)`) | `min_kv_seq_len // 256 &ge; BLOCK_N * 3 = 384` | `min_kv_seq_len // 256 &ge; BLOCK_N * 3 = 192` |
| Stage-2 reduce | skipped when `NUM_KV_SPLITS == 1` | always runs | always runs |

**`bh64` NUM_KV_SPLITS selection** (NUM_XCDS=8, BLOCK_H=64):

| batch | nhead | NUM_KV_SPLITS | min_kv_seq_len bound |
|-------|-------|---------------|----------------------|
| 64    | 64    | 4             | > 784                |
| 64    | 128   | 2             | > 388                |
| 128   | 64    | 2             | > 388                |
| 128   | 128   | 1             | > 193                |
| 256   | 64    | 1             | > 193                |
| 256   | 128   | 1             | > 193                |

**Page table modes** (`use_2d_view`, both regimes):
- `True`: `page_table = block_table [batch, max_seqlen]`, `seq_info = cache_seqlens [batch]`. Use for fixed-length or pre-padded variable-length sequences.
- `False`: `page_table = kv_indices [total_kv]`, `seq_info = kv_indptr [batch+1]`. Use for variable-length sequences without block_table construction.

**KV layout** (both regimes): By default `kv_c` is a flat `[N, 576]` buffer containing both the compressed latent (columns `[0, 512)`) and rope PE (columns `[512, 576)`). The kernel adds `kv_pe_offset` to k_pe column offsets — set to `kv_lora_rank` (512) when `k_pe` shares `kv_c` (default), or `0` when `k_pe` is a separate buffer. The kernel auto-selects the load instruction via `WITHIN_2GB`: `buffer_load_to_shared` (scalar base + 32-bit offsets) when KV caches &le; 2 GB, or `global_load_to_shared` (64-bit pointer tensors) when KV caches > 2 GB.

**`bh64` perf** (MI350, ctx=16384, bf16 Q + bf16 KV; compute-bound):

```
python op_tests/test_mla.py -c 16384 -b 64 128 -n 64,1 128,1 -d bf16 -kvd bf16
```

| batch | nhead | ASM TFLOPS | Gluon TFLOPS | Speedup |
|-------|-------|------------|--------------|---------|
| 64    | 64    | 350.1      | 453.6        | 1.30&times; |
| 128   | 64    | 368.1      | 462.0        | 1.26&times; |
| 64    | 128   | 469.9      | 529.3        | 1.13&times; |
| 128   | 128   | 476.7      | 563.0        | 1.18&times; |

**`bh16bn128` perf** (MI350, ctx=10M, bf16 Q + fp8 KV; memory-bound):

```
python op_tests/test_mla.py -c 10000000 -b 1 -n 16,1 -d bf16 -kvd fp8
```

| batch | nhead | ASM TB/s | Gluon TB/s | Speedup |
|-------|-------|----------|------------|---------|
| 1     | 16    | —        | 4.58       | —       |

ASM does not support this regime (bf16 Q + fp8 KV → "don't support this case"). Gluon reaches ~70% of MI350's 6.5 TB/s HBM peak (wall-clock 1256 &mu;s).

**`bh16bn64` perf** (MI350, ctx=10M, bf16 Q + bf16 KV; memory-bound):

```
python op_tests/test_mla.py -c 10000000 -b 1 -n 16,1 -d bf16 -kvd bf16
```

| batch | nhead | ASM TB/s | Gluon TB/s | Speedup |
|-------|-------|----------|------------|---------|
| 1     | 16    | 0.69     | 5.33       | 7.71&times; |

Gluon reaches ~82% of MI350's 6.5 TB/s HBM peak (wall-clock 2162 &mu;s vs ASM 16659 &mu;s).

### `pa_decode_gluon.py` — Paged Attention Decode

**Function:** `pa_decode_gluon(output, query, key_cache, value_cache, context_lengths, block_tables, softmax_scale, query_length, max_context_partition_num, context_partition_size, compute_type, query_scale, key_scale, value_scale, ...)`

**Description:** Paged attention decode with partitioned KV (first pass + reduction). Supports MTP (multi-token prefill, query_length &le; 4), sliding window, ALiBi, causal masking. Three inner kernel variants for different KV block sizes.

| Parameter | Details |
|-----------|---------|
| Arch | gfx942 (CDNA3) and gfx950 (CDNA4) |
| Q dtype | fp8_e4m3fnuz, bf16, fp16 |
| KV dtype | fp8_e4m3fnuz, bf16, fp16 |
| Output | bf16 (fp8 mode), or matches compute_type |
| KV block sizes | 16, 64, 1024 (selected by kernel variant) |
| Context partition | 256 (static_assert) |
| Constraint | `query_length * query_group_size` &le; 64 |

### `sparse_attention_dsv4.py` — DeepSeek V4 Sparse MLA (Prefill + Decode)

**Functions:** `sparse_attn_prefill(q, kv, indices, topk_length, ...)`, `sparse_attn_decode(q, kv_cache, swa_k_cache, swa_only, topk_indices, topk_lens, swa_indices, ...)`

**Description:** DeepSeek V4 Compressed Sparse Attention. Each query token attends to a sparse, per-token set of KV positions (top-k / sliding-window) supplied as CSR `(indices, indptr)` with `-1` sentinels masked out :

- **Prefill** (`_sparse_attn_prefill_kernel`): bf16 KV in a flat `[N, D]` buffer, ragged per-query indices, optional per-head attention sink. One program per query token &times; `BLOCK_H` head block. The gathered KV tile is DMA'd global&rarr;LDS once via `async_copy` and read in both orientations (K^T for QK, V for PV) through a `permute` view of a single padded LDS buffer.
- **Decode** (`_sparse_attn_decode_kernel`): fp8_ds_mla paged cache, with two parallel sparse passes — a "main"/sliding-window pass over `swa_k_cache` and an optional "extra"/top-k pass over `kv_cache` — that share one running softmax statistic. The NoPE half is dequantized from FP8 (per-64-element exp scale), the RoPE half is bf16.

Ported from vLLM's `rocm_aiter_mla_sparse`. The host wrapper (`aiter/ops/triton/attention/sparse_attention_dsv4.py`) selects the Gluon kernel on gfx950 when the KV pool / paged cache fits within the 2 GB buffer-descriptor cap, otherwise falls back to the Triton kernel (whose 32-bit buffer offsets carry the same cap).

| Parameter | Details |
|-----------|---------|
| Arch | gfx950 (CDNA4) only |
| Q dtype | bf16 |
| KV dtype | bf16 flat buffer (prefill); fp8_ds_mla `uint8` paged cache (decode) |
| Output | bf16 |
| head_dim | 512 = nope_head_dim (448) + rope_head_dim (64), enforced |
| Decode cache row | 576 B (448 FP8 NoPE + 64 bf16 RoPE) + 8 B per-token scale = 584 B |
| Sparsity | CSR ragged `(indices, indptr)`; `-1` sentinels masked |
| Attn sink | optional `[H]` fp32 per-head softmax-denominator bias |
| MFMA | QK 16&times;16&times;32 (kWidth=8), PV 16&times;16&times;16 (kWidth=4) |
| Tunable | BLOCK_H, BLOCK_K, num_warps, waves_per_eu (`@triton.autotune`) |

**Decode passes** (`swa_only`):
- `True`: only the main/SWA pass over `swa_k_cache` runs (`HAS_EXTRA=False`).
- `False`: the extra/top-k pass over `kv_cache` also runs, sharing the same online-softmax accumulator.

**Test** (prefill accuracy + perf vs a torch reference, gfx950):

```
# config is num_queries,num_heads,num_kv,topk
python op_tests/test_sparse_attention_dsv4.py --cfgs 4096,128,4096,512

# multiple shapes; add --sink to exercise the per-head attention-sink path
python op_tests/test_sparse_attention_dsv4.py \
    --cfgs 4096,128,4096,512 8192,128,8192,1024 --sink
```

Per-query ragged top-k indices keep a random count in `[topk//4, topk]` of
unique slots; the test checks `checkAllclose` against a per-query masked-softmax
torch reference and reports TFLOPS / GB/s.

**Perf** (MI350, prefill, bf16, H=128):

```
python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py --shapes prefill
```

| Q | Kv | topk | Triton TFLOPS | Gluon TFLOPS | Speedup |
|------|------|------|---------------|--------------|---------|
| 4096 | 4096 | 512  | 192.2 | 335.0 | 1.74&times; |
| 4096 | 4096 | 1024 | 209.0 | 383.1 | 1.83&times; |
| 8192 | 8192 | 512  | 191.0 | 329.8 | 1.73&times; |
| 8192 | 8192 | 1024 | 211.2 | 369.8 | 1.75&times; |

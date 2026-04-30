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
  <td><code>mla_decode_gluon</code></td><td>MLA<br>Decode</td><td>CDNA4</td>
  <td nowrap>Q: bf16, KV: bf16, Out: bf16<br>batch_size in {64, 128, 256}<br>nhead in {64, 128}<br>PAGE_SIZE=1, BLOCK_H=BLOCK_N=64<br>min_kv_seq_len &gt; NUM_KV_SPLITS&times;(192+NUM_KV_SPLITS)<br>(NUM_KV_SPLITS auto-picked &isin; {1,2,4})</td>
  <td>python op_tests/test_mla.py \<br>-c 10000 -b 128 -n 128,1 \<br>-d bf16 -kvd bf16</td>
  <td>~570<br>TFLOPS</td><td>~480<br>TFLOPS</td><td>—</td>
</tr>
<tr>
  <td><code>pa_decode_gluon</code></td><td>Paged Attn<br>Decode</td><td>CDNA3<br>CDNA4</td>
  <td nowrap>Q: fp8/bf16/fp16<br>KV: fp8/bf16/fp16<br>Out: bf16 or match<br>query_len &le; 4<br>query_len &times; group_size &le; 64<br>ctx_partition = 256</td>
  <td>python op_tests/triton_tests/<br>test_pa_decode_gluon.py</td>
  <td>TBD</td><td>TBD</td><td>TBD</td>
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

**Function:** `mla_decode_gluon(q_nope, q_pe, kv_c, o, page_table, seq_info, sm_scale, k_pe=None, kv_pe_offset=512, use_2d_view=True, min_kv_seq_len=1)`

**Description:** Multi-head Latent Attention (DeepSeek MLA) decode kernel with split-KV. Q is split into compressed latent (`q_nope`, dim=kv_lora_rank) and rope positional encoding (`q_pe`, dim=qk_rope_head_dim). KV cache is a flat `[N, 576]` buffer (`kv_c`). Uses 3-stage async copy pipeline with double-buffered page numbers and KV tiles.

The wrapper auto-picks `NUM_KV_SPLITS &isin; {1, 2, 4}` so the launch fills ~256 workgroups (one wave on MI350). When `NUM_KV_SPLITS == 1`, stage-1 writes the final attention output directly to `o` (no temp buffer, no reduce). When `NUM_KV_SPLITS > 1`, stage-1 writes per-split `(acc, lse)` to a temp buffer and a stage-2 Triton kernel (`_mla_softmax_reducev_kernel`) combines them into `o`.

Modified from [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).

| Parameter | Details |
|-----------|---------|
| Arch | gfx950 (CDNA4) only |
| Q dtype | bf16 only (static_assert) |
| KV dtype | bf16 only (static_assert) |
| Output | bf16 |
| batch_size | 64, 128, or 256 only |
| nhead | 64 or 128 only |
| Page size | 1 only (static_assert) |
| Block sizes | BLOCK_H=64 (heads), BLOCK_N=64 (KV seq) — fixed |
| MFMA | 16&times;16&times;32, warps=[4,1] |
| NUM_KV_SPLITS | auto-picked &isin; {1, 2, 4} from (batch, nhead) to target ~256 workgroups |
| Seq constraint | `min_kv_seq_len > NUM_KV_SPLITS * (PIPELINE_STAGES * BLOCK_N + NUM_KV_SPLITS)` with `PIPELINE_STAGES=3` (per-split `num_iter > 3`) |

**NUM_KV_SPLITS selection** (NUM_XCDS=8, BLOCK_H=64):

| batch | nhead | NUM_KV_SPLITS | min_kv_seq_len bound |
|-------|-------|---------------|----------------------|
| 64    | 64    | 4             | > 784                |
| 64    | 128   | 2             | > 388                |
| 128   | 64    | 2             | > 388                |
| 128   | 128   | 1             | > 193                |
| 256   | 64    | 1             | > 193                |
| 256   | 128   | 1             | > 193                |

**Page table modes** (`use_2d_view`):
- `True`: `page_table = block_table [batch, max_seqlen]`, `seq_info = cache_seqlens [batch]`. Use for fixed-length or pre-padded variable-length sequences.
- `False`: `page_table = kv_indices [total_kv]`, `seq_info = kv_indptr [batch+1]`. Use for variable-length sequences without block_table construction.

**KV layout**: By default `kv_c` is a flat `[N, 576]` buffer containing both the compressed latent (columns `[0, 512)`) and rope PE (columns `[512, 576)`). The kernel adds `kv_pe_offset` to k_pe column offsets — set to `kv_lora_rank` (512) when `k_pe` shares `kv_c` (default), or `0` when `k_pe` is a separate buffer. The kernel auto-selects the load instruction via `WITHIN_2GB`: `buffer_load_to_shared` (scalar base + 32-bit offsets) when KV caches &le; 2 GB, or `global_load_to_shared` (64-bit pointer tensors) when KV caches > 2 GB.

**Perf** (MI350, ctx=16384, bf16):

```
python op_tests/test_mla.py -c 16384 -b 128 -n 64,1 128,1 -d bf16 -kvd bf16
```

| batch | nhead | ASM TFLOPS | Gluon TFLOPS | Speedup |
|-------|-------|------------|--------------|---------|
| 64    | 64    | 367.0      | 466.6        | 1.27&times; |
| 128   | 64    | 379.0      | 485.3        | 1.28&times; |
| 64    | 128   | 476.9      | 543.2        | 1.14&times; |
| 128   | 128   | 490.2      | 578.9        | 1.18&times; |

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

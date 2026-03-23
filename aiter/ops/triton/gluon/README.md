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
  <td nowrap>Q: bf16, KV: bf16, Out: bf16<br>PAGE_SIZE=1, BLOCK_H=64<br>seq_len &gt; 192<br>nhead % 64 == 0<br>KV buf &le; 4 GB for zero-copy</td>
  <td>python op_tests/test_mla.py<br>-c 4096 -b 128 -n 128,1<br>-d bf16 -kvd bf16</td>
  <td>~530<br>TFLOPS</td><td>~435<br>TFLOPS</td><td>—</td>
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

**Function:** `mla_decode_gluon(q_nope, q_pe, kv_c, k_pe, o, page_table, seq_info, sm_scale, kv_pe_offset=0, use_2d_view=True)`

**Description:** Multi-head Latent Attention (DeepSeek MLA) decode kernel. Single-stage (no split-K). Q is split into compressed latent (`q_nope`, dim=kv_lora_rank) and rope positional encoding (`q_pe`, dim=qk_rope_head_dim). KV cache similarly has `kv_c` and `k_pe` components. Uses 3-stage async copy pipeline with double-buffered page numbers and KV tiles.

Modified from [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).

| Parameter | Details |
|-----------|---------|
| Arch | gfx950 (CDNA4) only |
| Q dtype | bf16 only (static_assert) |
| KV dtype | bf16 only (static_assert) |
| Output | bf16 |
| Page size | 1 only (static_assert) |
| Block sizes | BLOCK_H=64 (heads), BLOCK_N=64 (KV seq) — fixed |
| MFMA | v4, 32&times;32&times;16, warps=[2,2] |
| Seq constraint | seq_len > 192 (kernel assumes `num_iter > 3`) |
| nhead | must be multiple of BLOCK_H=64 (tested: 128) |

**Page table modes** (`use_2d_view`):
- `True`: `page_table = block_table [batch, max_seqlen]`, `seq_info = cache_seqlens [batch]`. Use for fixed-length or pre-padded variable-length sequences.
- `False`: `page_table = kv_indices [total_kv]`, `seq_info = kv_indptr [batch+1]`. Use for variable-length sequences without block_table construction.

**Zero-copy KV path** (`kv_pe_offset > 0`): Pass the same `[N, 576]` buffer as both `kv_c` and `k_pe`. The kernel adds `kv_pe_offset` to k_pe column offsets. Requires buffer size &le; 4 GB (Triton buffer_load `num_records` limit). For larger buffers, fall back to contiguous splits.

**Perf** (gfx950, batch=128, ctx=4096, nhead=128, bf16):
- Gluon: ~275 us, ~530 TFLOPS
- ASM baseline: ~335 us, ~435 TFLOPS

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


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
  <td><code>gemm_a8w8_blockscale</code></td><td>GEMM<br>(block-scale)</td><td>CDNA4</td>
  <td nowrap>A/B: fp8_e4m3 (mfma_scaled)<br>Out: bf16/fp16<br>Per-tile scales:<br>A [M, K/GROUP_K],<br>B [N/GROUP_N, K/GROUP_K]<br>BLOCK_K=128, NUM_WARPS=4<br>(BM,BN) &isin; {(64,128),<br>(128,128),(128,256)}</td>
  <td>python op_tests/op_benchmarks/<br>triton/bench_gemm_a8w8_<br>blockscale.py -gluon</td>
  <td>~1271<br>TFLOPS<br>(4Kx4Kx4K)</td><td>—</td><td>TBD</td>
</tr>
<tr>
  <td rowspan="5"><code>mla_gluon</code></td><td rowspan="5">MLA</td><td rowspan="5">CDNA4</td>
  <td rowspan="2" nowrap>(bh64)<br>Q: bf16, KV: bf16, Out: bf16<br>batch_size in {64, 128, 256}<br>nhead in {64, 128}<br>PAGE_SIZE=1<br>BLOCK_H=BLOCK_N=64</td>
  <td>python op_tests/test_mla.py \<br>-c 16384 -b 64 128 \<br>-n 64,1 128,1 \<br>-d bf16 -kvd bf16</td>
  <td>~563<br>TFLOPS</td><td>~477<br>TFLOPS</td><td>—</td>
</tr>
<tr>
  <td>python op_tests/op_benchmarks/<br>triton/bench_sparse_attention_dsv4.py \<br>--prefill_cfgs 4096,128,4096,1024<br>(sparse prefill)</td>
  <td>~507<br>TFLOPS</td><td>—</td><td>—</td>
</tr>
<tr>
  <td nowrap>(bh16bn128)<br>Q: bf16, KV: fp8, Out: bf16<br>batch_size = 1<br>nhead &le; 16<br>PAGE_SIZE=1<br>BLOCK_H=16, BLOCK_N=128</td>
  <td>python op_tests/test_mla.py \<br>-c 10000000 -b 1 -n 16,1 \<br>-d bf16 -kvd fp8</td>
  <td>~4.58<br>TB/s</td><td>—</td><td>—</td>
</tr>
<tr>
  <td rowspan="2" nowrap>(bh16bn64)<br>Q: bf16, KV: bf16<br>Out: bf16 (+fp32 lse<br>with -lse)<br>nhead &le; 16<br>batch_size &ge; 1<br>NUM_KV_SPLITS=<br>max(1,min(256//B,<br>cdiv(seq,64)))<br>(B*splits &le; 256)<br>PAGE_SIZE=1<br>BLOCK_H=16, BLOCK_N=64</td>
  <td>python op_tests/test_mla.py \<br>-c 10000000 -b 1 -n 16,1 \<br>-d bf16 -kvd bf16<br>(full decode)</td>
  <td>~5.33<br>TB/s</td><td>~0.69<br>TB/s</td><td>—</td>
</tr>
<tr>
  <td>python op_tests/test_mla.py \<br>-c 100000 -b 4 -n 16,1 \<br>-d bf16 -kvd bf16 \<br>-lse<br>(full decode + lse)</td>
  <td>~4.31<br>TB/s</td><td>—</td><td>—</td>
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

### `gemm_a8w8_blockscale.py` — FP8 GEMM with block-scale quantization

**Function:** `gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=bf16, y=None, config=None)`

**Description:** Y = X &times; W^T where X and W are fp8_e4m3 and each carries
per-tile (block) fp32 scales — X by `[M, ceil(K/GROUP_K)]` and W by
`[ceil(N/GROUP_N), ceil(K/GROUP_K)]`. The inner instruction is
`gl.amd.cdna4.mfma_scaled` (CDNA4 V_MFMA_SCALE_F32_*) which folds both scales
into the dot product. The kernel pipelines two independent async-copy streams
— A/B operands and the per-tile scales — through `NUM_STAGES`-deep LDS
multi-buffers; pipelining scales separately is the main perf delta vs. the
equivalent Triton kernel.

**Pipeline.** Three things are in flight on every main-loop iter `k`:

| Stream | Operation | K-tile index |
|--------|-----------|--------------|
| global -> LDS | prefetch A, B (`async_copy.buffer_load_to_shared`) | `k + 2` |
| global -> LDS | prefetch a_scale, b_scale (separate stream) | `k + 1` |
| LDS -> regs | load operands into mfma dot layouts (becomes next iter's `prev_*`) | `k + 1` |
| LDS -> regs | load scales (broadcast across MFMA lanes) | `k` |
| compute | `mfma_scaled` on `prev_a / prev_b` (LDS-read in iter `k - 1`) | `k` |
| compute | scale-accumulate `acc += mfma_out * a_scale * b_scale` | `k` |

The body of the loop is hard-coded as `EVEN_K=True` so the compiler drops the
K-mask branch from the hot path. A `commit_group()` is issued *before* the
LDS reads / MFMA so the backend can hoist `buffer_load_to_shared` up past
the `ds_read_b128` + `v_mfma_*`; deferring the commit serializes the global
load behind compute and tanks throughput.

A statically-unrolled wind-down (1 iter when `EVEN_K`, 2 iters when not — the
extra iter covers the boundary-masked last tile) drains the pipe. The unroll
is what kills the `prev_a / prev_b` PHI node that would otherwise force the
dot operands out of AGPRs in the hot loop. Runtime `num_k_iter > N` guards
make the wind-down a no-op for small-K shapes so only the Final iter runs.

**Parameters**

| Parameter | Details |
|-----------|---------|
| Arch | gfx950 (CDNA4) only |
| A / B dtype | fp8_e4m3 |
| Output | bf16 (default), fp16 |
| Scales | fp32 |
| Scale tile | `GROUP_K`, `GROUP_N` (powers of two, inferred from `w_scale.shape`) |
| Tile shapes | `(BLOCK_SIZE_M, BLOCK_SIZE_N)` &isin; `{(64,128), (128,128), (128,256)}` |
| Baked-in | `BLOCK_SIZE_K = 128`, `NUM_WARPS = 4`, MFMA `16&times;16&times;128` |
| SplitK | `NUM_KSPLIT` (separate reduce kernel `_gemm_a8w8_blockscale_reduce_kernel`) |
| Tunable | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `NUM_STAGES`, `NUM_XCDS` |
| Config | `$AITER_TRITON_CONFIGS_PATH/gemm/gluon/gfx950-GEMM-A8W8_BLOCKSCALE[-N=*-K=*].json` |

**Perf** (MI350, `-gluon` flag selects this kernel; vs. the in-tree Triton kernel):

```
python op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py [-gluon]
```

| M | N | K | Gluon TFLOPS | Triton TFLOPS | Speedup |
|---|---|---|--------------|---------------|---------|
| 128   | 1280 | 8192 | 100.6  | 51.1   | 1.97&times; |
| 2048  | 1280 | 8192 | 677.3  | 341.7  | 1.98&times; |
| 4096  | 1280 | 8192 | 863.5  | 670.9  | 1.29&times; |
| 8192  | 1280 | 8192 | 887.1  | 683.1  | 1.30&times; |
| 16384 | 1280 | 8192 | 1164.9 | 899.0  | 1.30&times; |
| 4096  | 4096 | 4096 | 1271.4 | 1013.2 | 1.26&times; |
| 4096  | 4096 | 4160 | 1076.1 | 862.7  | 1.25&times; |

(Small-M shapes where the kernel is launch- / occupancy-bound — e.g. `M=192` and `M=512` at `N=1280, K=8192` — are currently slower than Triton; tuning continues.)

---

## Attention Kernels

### `mla_gluon.py` — MLA Decode + DeepSeek V4 Sparse Prefill

**Function:** `mla_gluon(q_nope, q_pe, kv_c, o, page_table, seq_info, sm_scale, k_pe=None, kv_pe_offset=512, use_2d_view=True, kv_scale=1.0, min_kv_seq_len=1, return_lse=False)`

**Description:** Multi-head Latent Attention (DeepSeek MLA) kernel with split-KV. For MLA Decode, Q is split into compressed latent (`q_nope`, dim=kv_lora_rank) and rope positional encoding (`q_pe`, dim=qk_rope_head_dim). KV cache is a flat `[N, 576]` buffer (`kv_c`). For DSv4 Sparse Prefill, Q packs compressed latent and positional encoding into one contiguous row (448 NoPE + 64 RoPE, `q_nope` with shape `[nquery, nhead, 512]`), KV cache has aligned `head_dim=512`, `q_pe` and `k_pe` can be left as placeholders. Uses 3-stage async copy pipeline with double-buffered page numbers and KV tiles.

The wrapper dispatches by `(nhead, kv_c.dtype)` to one of three compile-time regimes (single `@gluon.jit` kernel, REGIME constexpr gates layouts and grid mapping):

- **`bh64`** (`nhead in {64, 128}`): bf16 KV, BLOCK_H=64, BLOCK_N=64, multi-batch + XCD-aware 3-D grid. `NUM_KV_SPLITS` auto-picked &isin; {1, 2, 4} so the launch fills ~256 workgroups (one wave on MI350). When `NUM_KV_SPLITS == 1`, stage-1 writes the final attention output directly to `o` (no temp buffer, no reduce). When `NUM_KV_SPLITS > 1`, stage-1 writes per-split `(acc, fp32 lse)` and stage-2 (`_mla_softmax_reducev_kernel`) reduces them into `o`.
- **`bh16bn128`** (`nhead &le; 16`, `batch_size == 1`, fp8 KV): BLOCK_H=16, BLOCK_N=128, 2-D grid `(1, NUM_KV_SPLITS)` with token-bound `NUM_KV_SPLITS = max(1, min(256, min_kv_seq_len))` — 256 for the normal long-context path, reduced only for small kv (`min_kv_seq_len < 256`) so every split stays non-empty. Optional `kv_scale` dequant. Stage-2 reduce runs whenever `NUM_KV_SPLITS > 1` (skipped via the fast path only at `min_kv_seq_len == 1`). Supports the general case `num_iter &isin; {1, 2, ...}` (no `gl.assume(num_iter >= 3)`). `NHEAD < BLOCK_H` masks OOB heads on Q load and O store (wasted MFMA lanes are free; this regime is memory-bound).
- **`bh16bn64`** (`nhead &le; 16`, bf16 KV): BLOCK_H=16, BLOCK_N=64, 2-D grid `(batch_size, NUM_KV_SPLITS)` with block-bound `NUM_KV_SPLITS = max(1, min(256 // batch_size, cdiv(min_kv_seq_len, BLOCK_N)))` — fills ~256 WGs but never splits a sequence into more than its 64-token block count, so small kv is supported and it collapses to 1 (one WG per batch over the whole sequence) when `min_kv_seq_len <= 64`. Use when KV is kept in bf16 (no fp8 quant). Same `NHEAD < BLOCK_H` masking. Full decode (stage-1, plus stage-2 reduce into `o` when `NUM_KV_SPLITS > 1`).

All three regimes run the full decode and dsv4 prefill. `return_lse=True` also returns the merged fp32 lse `[batch, nhead]`, so `mla_gluon(...)` returns `(o, final_lse)` instead of `(o, None)`.

Modified from [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).

| Parameter | `bh64` regime | `bh16bn128` regime | `bh16bn64` regime |
|-----------|---------------|--------------------|--------------------|
| Arch | gfx950 (CDNA4) | gfx950 (CDNA4) | gfx950 (CDNA4) |
| Q dtype | bf16 | bf16 | bf16 |
| KV dtype | bf16 | fp8 | bf16 |
| Output | bf16 | bf16 | bf16 |
| batch_size | 64, 128, or 256 | 1 | &ge; 1 |
| nhead | 64 or 128 | &le; 16 (tested: 4, 8, 16) | &le; 16 (tested: 4, 8, 16) |
| Page size | 1 | 1 | 1 |
| BLOCK_H | 64 | 16 | 16 |
| BLOCK_N | 64 | 128 | 64 |
| MFMA | 16&times;16&times;32, warps=[4,1] | 16&times;16&times;32, warps=[1,4] | 16&times;16&times;32, warps=[1,4] |
| Grid | 3-D XCD-aware | 2-D `(1, NUM_KV_SPLITS)` | 2-D `(batch, NUM_KV_SPLITS)` |
| NUM_KV_SPLITS | auto &isin; {1, 2, 4} from (batch, nhead) | `max(1, min(256, min_kv_seq_len))` (token-bound; 256 for ctx &ge; 256) | `max(1, min(256 // batch_size, cdiv(min_kv_seq_len, 64)))` (block-bound; collapses to 1 for ctx &le; 64) |
| `kv_scale` | unused (pass 1.0) | dequant scale folded into `qk_scale` (applied before softmax for fp8 correctness) | unused (pass 1.0) |
| Seq constraint | `min_kv_seq_len > NUM_KV_SPLITS * (3 * BLOCK_N + NUM_KV_SPLITS)` (the `3` matches the kernel's `gl.assume(num_iter > 3)`) | `min_kv_seq_len &ge; 1` (small kv 1..256 supported; token-bound clamp keeps splits non-empty) | `min_kv_seq_len &ge; 1` (small kv 1..256 supported; block-bound clamp keeps splits non-empty) |
| Stage-2 reduce | skipped when `NUM_KV_SPLITS == 1` | skipped when `NUM_KV_SPLITS == 1` (i.e. `min_kv_seq_len == 1`) | skipped when `NUM_KV_SPLITS == 1` |

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

**`return_lse` perf** (MI350, bf16 Q + bf16 KV; memory-bound; full decode + lse):

```
python op_tests/test_mla.py -c 10000 100000 -b 1 3 4 -n 16,1 -d bf16 -kvd bf16 -lse
```

| ctx_lens | batch | NUM_KV_SPLITS | num_iter / split | us | TB/s |
|----------|-------|---------------|------------------|-----|------|
| 10K      | 1     | 157           | 1                | 39.09  | 0.30 |
| 10K      | 3     | 85            | 2                | 32.33  | 1.07 |
| 10K      | 4     | 64            | 3                | 26.35  | 1.75 |
| 100K     | 1     | 256           | 7                | 63.78  | 1.81 |
| 100K     | 3     | 85            | 19               | 88.77  | 3.89 |
| 100K     | 4     | 64            | 25               | 106.96 | 4.31 |

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

# gfx1250 QH128 MLA: page1 and shuffled page64

Two directly callable kernels let the framework choose the KV cache layout
using model-level measurements. Both consume FP8 E4M3FN Q/KV and compute one
query per sequence, 128 query heads, one KV head, QK width 576 and value width
512. Neither splits KV or returns partial logits. Both write preallocated
BF16 output through the shared `module_mla_asm` / `asm_mla.cu` module.

| | Page1 | Shuffled page64 |
|---|---|---|
| Entry point in `aiter.mla` | `mla_decode_fwd_ps1_qh128_asm` | `mla_decode_fwd_ps64_qh128_asm` |
| KV shape | `[physical_tokens,1,1,576]` | `[physical_pages,1,64,576]` |
| KV bytes | Token-major: 512 NoPE/V values, then 64 RoPE values | Separate NoPE/RoPE planes, each tiled into 16-token × 16-dimension blocks |
| Metadata | Contiguous int32 `kv_indptr[B+1]`, `kv_indices[nnz]` | Int32 `seq_lens[B]`, dense `page_table[B,max_pages]` |
| FP32 natural-log LSE | Required preallocated `[B,128]` buffer | Optional preallocated `[B,128]` buffer |
| Native kernel arguments | 80 bytes | 88 bytes |

Q is `[B,128,576]`, with dense batches and head stride 576 or 768 bytes. Each
Q/KV descale is one device FP32 scalar; pass `None` for a cached scalar one.
Used Q/KV values and scales must be finite. Empty sequences write O=0 and
LSE=-inf. Warm up on the target device before capturing a graph. Launches use
the current stream and do not copy metadata to the CPU.

Page1 CSR contents are a caller contract: indptr must be nonnegative,
monotonic and within the indices allocation; used indices must be valid
physical token IDs. Repeated/shared IDs are allowed. The page1 kernel always
writes LSE; reuse a scratch LSE buffer if the framework discards it. Page64
retains its device-side invalid-length/page-ID handling. A shape check cannot
distinguish shuffled bytes from unshuffled bytes in an identically shaped
tensor: the cache writer and decoder must agree on the layout.

```python
from aiter.mla import mla_decode_fwd_ps1_qh128_asm, mla_decode_fwd_ps64_qh128_asm

# Allocate out/lse and prepare metadata outside a timed/captured launch.
mla_decode_fwd_ps1_qh128_asm(
    q, token_major_kv, kv_indptr, kv_indices, out,
    q_scale, kv_scale, softmax_scale, lse=lse,
)
mla_decode_fwd_ps64_qh128_asm(
    q, shuffled_kv, seq_lens, page_table, out,
    q_scale, kv_scale, softmax_scale, lse=lse,
)
```

These are explicit choices. The general `mla_decode_fwd` dispatcher does not
automatically select either new specialization. The manifest's explicit
`page_size=1/64` rows are excluded from legacy launchers with different ABIs.

## Why DeepSeek R1 needs a framework-level comparison

The following evidence is from the inspected ATOM revision
[`2655fd25dee04ff27a8c2026d199495b99718dbc`](https://github.com/ROCm/ATOM/tree/2655fd25dee04ff27a8c2026d199495b99718dbc).
It establishes distinct cache preparation paths, not a measured end-to-end
winner for these two kernels.

- The [DeepSeek R1 recipe](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/recipes/DeepSeek-R1.md#L1-L38)
  describes MLA, FP8 KV cache, TP8 and optional MTP. The model constructs
  [the MLA attention layer from the local query heads and the latent+RoPE cache](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/models/deepseek_v2.py#L2547-L2589).
- [`ATOM_MLA_PAGE_SIZE` defaults to 1](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/utils/envs.py#L98).
  The [plain path uses interleaved per-token cache writers](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attention_mla.py#L654-L684).
  A [scheduler allocation block is distinct from an attention page](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attentions/mla_kv_pool.py#L10-L21);
  page1 does not imply a separate physical allocation per token.
- ATOM has a [shuffled block64 decode branch](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attention_mla.py#L2136-L2159),
  selected by `ATOM_USE_TRITON_MLA` and `ATOM_USE_TRITON_MLA_SHUFFLE_KV`.
  Prefill uses [a shuffle-aware cache writer](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attention_mla.py#L2516-L2553).
  Decode can [fuse RoPE, cache insertion and layout handling](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attention_mla.py#L2652-L2720).
  Thus the layout must already be correct before decode, but it does not
  necessarily require a separate full-cache shuffle on every decode step.
  [`_shuffled_kv_view` is only a view](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attention_mla.py#L2010-L2029);
  timing that view does not measure the layout preparation work.
- The [context-row KV fusion gate](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/model_ops/attention_mla.py#L686-L705)
  excludes segmented and shuffled caches. Selecting a layout can therefore
  change which upstream fusions are available.

Measure cache write/quantization/layout preparation plus decode, then the
complete attention/model step, with identical workloads and output contracts.
If converting an existing plain cache, include that conversion at its actual
frequency. If inserting directly into a shuffled cache, measure the fused
writer's cost instead of charging an artificial full-cache repack each step.
Include page-table/CSR preparation where it is actually performed. Record GPU
clocks and compare prefill and decode separately before using serving metrics
such as TPOT/throughput to choose the framework default.

Three integration details matter for R1:

1. QH128 means **128 heads in this kernel invocation**. ATOM
   [divides model heads by TP size](https://github.com/ROCm/ATOM/blob/2655fd25dee04ff27a8c2026d199495b99718dbc/atom/models/deepseek_v2.py#L2337-L2345);
   DCP/query replication may change the decode head set. A model's total head
   count alone does not make the specialization applicable.
2. Both kernels handle one query per sequence. MTP/multi-query or split-K
   paths need their existing kernels or a separate integration; do not route
   them solely by page size.
3. ATOM's current shuffled branch passes BF16 Q and no Q descale, as noted in
   its decode call. These ASM entry points require FP8 Q. A framework adapter
   must supply the matching Q quantization and scale and include its cost.
   Also, ATOM's non-Triton `ATOM_MLA_PAGE_SIZE=64` **segmented** layout is a
   different layout from the Gluon-shuffled page64 bytes required here.

## Shared validation and kernel timing

Both implementations use `op_tests/test_mla_decode_pagesize64.py`, including
the existing FP32 reference and performance reporting:

```bash
python -m pytest op_tests/test_mla_decode_pagesize64.py -q
python op_tests/test_mla_decode_pagesize64.py --page-size 1 -n 128,1 --split-kv 1 -b 4 -c 65 1024
python op_tests/test_mla_decode_pagesize64.py --page-size 64 --kv-shuffled -n 128,1 --split-kv 1 -b 4 -c 65 1024
```

CLI timings exclude input generation and KV packing. Both CLI paths write
O+LSE, allowing the same output contract; they do not establish model-level
performance. The kernels use FP8-rounded probabilities and delayed softmax
rescaling, so tests compare against FP32 attention with the stated tolerance,
not bitwise equality between the two algorithms.

The page1 CO is built from `mla_v3_qh128_ps1.sp3`, source SHA256
`662da95b53c1bf03e6b72ad043ef25b9a4651be9a44983c7fa988dcd9d45a88f`.
Rebuilding with the existing `--raw-qk-cover` profile reproduces the original
CO `6f337d634fe1afc38028f15c1e6133ab2806d20c94c8a03329277acf0160eddc` byte for byte.
The packaged CO uses the AITER export symbol and has SHA256
`09c7c911eca85e1e35e9261fd299a4a9ac30b5490b0c36391dc8a0e338e625be`;
its instruction image is unchanged. The page64 CO is unchanged by this addition.

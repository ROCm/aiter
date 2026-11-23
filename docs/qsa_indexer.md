# QSA indexer on gfx942

The QSA indexer selects request-local token IDs for sparse attention. It uses
four index-query heads, one index-key head, head dimension 128, compression
ratio 4, and at most 512 complete key groups per query. The output has 2051
columns: selected four-token groups, up to three causal tail tokens, then -1.
Selected groups are ordered by group ID. Equal scores keep the lower IDs.

The implementation is ported from [PyHIP #57](https://github.com/tingqli/pyhip/pull/57)
and [#62](https://github.com/tingqli/pyhip/pull/62), revision
`8027faad29415747707670db7776528c6b72db81`, under the MIT license.
It is validated on MI308X (gfx942). It is an inference operator.

## Operator boundary

Import the public functions from `aiter.ops.qsa`:

```python
from aiter.ops.qsa import (
    qsa_decode_indexer,
    qsa_prefill_indexer,
    qsa_select_paged_tokens,
    qsa_sparse_prefill,
)
```

The complete indexers start at the projected BF16 `qk[rows,640]` tensor.
Each row contains four D128 Q heads followed by one raw D128 K head.
They perform Gemma RMSNorm, RoPE, raw-state writes, four-key compression,
compressed-state writes, logits, top-k, and token expansion.

The caller owns the projection, cache allocation, request metadata, TP
communication, and speculative acceptance. Main-attention Q/K/V are separate
D256 tensors. Index Q from `q_out` must not replace main-attention Q.

`qsa_select_paged_tokens` also exposes the selection stage separately. It reads
prepared index Q and compressed keys and does not update either cache. This
entry permits a framework to retain its own paged raw-key cache and compression.

## Norm and RoPE, including YaRN

Gemma RMSNorm applies `(1 + weight)`. A compressed key is the mean of four
raw keys, followed by K norm and RoPE at the first member's coordinates.
The implementation preserves the original BF16 intermediate roundings.

Complete entries consume `cos_sin_cache[N,64]`: the first 32 columns hold cos,
and the last 32 hold sin. `axis_map[32]` chooses coordinate 0, 1, or 2 for each
rotary pair. `positions` is int64 `[rows]` or `[3,rows]`, with unit token stride.
Logical token positions are separate arguments.

The caller supplies the final table, including YaRN frequency and magnitude
scaling. The indexer does not reconstruct it or test the RoPE Python class.
BF16 and FP32 tables are accepted; both follow the original BF16 RoPE arithmetic.
A framework's native FP32-table path can use different rounding and must be
validated separately. Q/K norm weights are BF16 `[128]`.

## Complete prefill

`qsa_prefill_indexer(qk, *, ...)` uses these additional inputs:

| Input | Contract |
|---|---|
| `query_lens`, `prefix_lens` | Nonnegative host integer sequences, one value per packed request. Prefixes are multiples of four. Their sums define the QK rows and final context lengths. |
| `logical_positions` | Contiguous int64 `[rows]`; request r, row i has position `prefix_lens[r]+i`. |
| `state_slots` | Int64 `[rows]`, raw-key/position state destinations. |
| `key_state`, `rope_state` | BF16 `[slots,1,128]` and int64 `[slots,3]`, updated in place. |
| `member_rows`, `write_locs` | Int64 and int32 `[groups]`. Each nonzero destination compresses four consecutive current QK rows starting at `member_rows[g]`. Destination zero is reserved. |
| `rope_matrix` | Int64 `[rows,3]`, the coordinates of each current row. |
| `compressed` | Contiguous BF16 `[compressed_slots,1,128]`, updated in place. |
| `token_slot_table` | Int32 `[requests,>=max_context]`, with unit column stride. Compressed key j is at `table[r,4*j]//4`. |

Optional `q_out` receives all prepared queries. Optional `out` is contiguous
int32 `[>=rows,2051]` and is returned unchanged as an object. A partial
`row_range=(first,end)` requires `out`; only those selection rows are written.
Prep and cache updates still cover every input row. An empty range is valid.
The caller performs any TP gather and removes extra frame-padding rows before
passing the result to sparse attention.

Prefill uses host layout planning. Capture requires a warm call on the same
stream with the same host lengths, range, and layout. A cold layout raises
before prep. Captured metadata stays pinned for the process lifetime so that
eager LRU eviction cannot free graph pointers. Changing sequence lengths during
replay requires the device-metadata decode entry instead. `torch.compile` is
not supported by the complete entries.

## Complete decode and multi-query windows

`qsa_decode_indexer(qk, *, ...)` has one metadata row per query token, including
padding. Its row count is not necessarily the number of requests.

| Input | Contract |
|---|---|
| `page_table` | Contiguous int32 `[rows,pages]`, one row per query. Page p contains compressed slots `16*p+[0,16)`. |
| `lengths` | Contiguous int32 `[rows]`, the number of causally visible compressed keys. |
| `logical_positions` | Int32 or int64 `[rows]`, request-local query positions. |
| `seq_lens` | GPU int32 `[rows]`, each query's sequence-length limit. |
| `group_locs` | Int32 `[rows,4]`, raw-state slots in oldest-to-newest order. |
| `write_locs` | Int32 `[rows]`, compressed destinations; zero is the reserved dump slot. |
| `multi_query_per_request` | False for independent request rows. True publishes all raw-state writes before a separate compression launch. |

The remaining state, norm, and RoPE arguments match prefill. Use eight raw-state
slots per request for windows of at most four queries. Reserved request slot
zero provides dump storage. The caller must ensure that live stores do not
overwrite group members before compression. Speculative accept/reject and
ring ownership are outside this operation.

Decode and multi-query mode support graph replay after warmup. Input addresses,
shapes, strides, and the mode remain fixed; lengths, positions, and page IDs can
change in place. Invalid rows must use safe dump addresses. Zero visible length
and a nonpositive causal window produce an all--1 selection.

Optional `q_out` receives BF16 `[rows,4,128]`. Optional `logits_out` is FP32
`[rows,16*pages]`, with 512 additional FP32 elements of backing storage after
the view. The index output is a fresh int32 `[rows,2051]` tensor for each eager
call; replay writes the output captured by its graph.

## Prepared-Q selection

Enable the new backend with `qsa_select_paged_tokens(..., allow_flydsl=True)`.
Its BF16 fast path takes Q `[rows,4,128]`, a compressed cache
`[pages,16,1,128]`, request page tables, `token_to_request`, logical positions,
and per-request context lengths. It reads the page table directly.

Optional host `query_lens` and `prefix_lens` enable the packed-prefill plan.
These prefixes need not be group-aligned because the caller already compressed
the keys. Static capture requires warmup on the same stream and layout.
Without host lengths, query batches of up to 256 rows use a device metadata
kernel and the decode scorer. They support changing metadata during replay.

The existing output-width rule and `logits_workspace_bytes` budget remain in
effect. Short output widths, other shapes, noncontiguous outputs, other page
sizes, and quantized compressed caches use the existing implementation.
The fast path refuses overlapping output/input storage.

## Limits and tests

The complete indexers support at most 262144 compressed keys per request
(1,048,576 tokens). The compressed pool must be smaller than 2 GiB. Decode
requires `(rows * width + 512) * 4 < 2**31`, where `width=16*pages`.
Complete prefill retains the original nominal 512 MiB logits budget and
1024-row minimum chunk; the widest rows use about 1.06 GiB. Prepared selection
uses smaller tiles when needed to honor its caller's logits budget.

All tensors must be on one GPU. Output buffers must not overlap inputs or cache
writes. Metadata values and cache contents must satisfy the documented layout;
shape checks do not validate every GPU-resident slot or position. A failed
launch must not be followed by a native retry over partially updated state.

The int32 `[rows,2051]` output is compatible with `qsa_sparse_prefill` at
T=512. That attention entry has its own limits, including an indices byte span
below 2 GiB; a 1M context normally uses smaller query chunks.

```bash
python op_tests/test_flydsl_qsa_indexer.py --interfaces full prepared --modes prefill decode verify -b 1
python -m pytest -q op_tests/test_flydsl_qsa_indexer.py -k dual
python -m pytest -q op_tests/triton_tests/attention/test_qsa_indexer.py
python op_tests/op_benchmarks/triton/bench_qsa_indexer.py
```

Tests cover original PyHIP cases, exact prepared Q/cache state, FP64 selection
boundaries, 16/32-bit chosen IDs, YaRN, graph replay, row ranges, workspace
budgets, and captured-plan lifetime. The standalone op test also prints a
correctness/performance table. Its timing method is separate from the original
[PyHIP benchmark protocol](https://github.com/tingqli/pyhip/blob/8027faad29415747707670db7776528c6b72db81/benchmarks/qsa/readme.md).

## Keeping both interfaces during a framework migration

The complete and prepared APIs are independent public contracts. The complete
entry owns preparation and state writes. Prepared selection consumes a caller's
Q/cache and must not update them. Both use the same selection kernels.

The `dual` tests start from the same QK, norm weights, RoPE table and logical
history. One branch runs the complete entry. The other receives Q and cache
contents from the independent reference in separate storage, with a different
physical page order. They compare Q, logical compressed groups and token
selections and check that prepared inputs remain unchanged. Short prepared
outputs are normalized to the full width only inside validation; the public
output-width rule is unchanged.

Decode tests advance actual and reference state separately for four steps,
including partial acceptance and ring wrap, then run normal decode from the
remaining state. Actual state is not reset to the reference between steps.
Both eager and graph paths cover one to four queries per request, padding,
MRoPE and YaRN. Other shared cases cover row ranges, wide IDs and 1M context.
These tests do not import a framework or claim to test its cache allocator.

`--interfaces full prepared` selects the timing boundaries and prints a separate
table for each. `full` measures QK through indices. `prepared` measures only
Q/cache through indices; reference prep, page conversion and buffer refresh are
outside timing. Both use `run_perftest`, with raw microseconds, dot-product
TFLOPS, logical TB/s and checked errors. Prepared correctness includes the
maximum per-row FP64 top-k boundary gap, with a tolerance of 1e-5.
The two times are not a same-boundary framework speedup comparison.

Changing a framework to the complete API also requires a compatible raw-state
contract. Current complete prep uses valid dump slots and the eight-slot ring;
prepared selection supports a caller-owned paged raw-state implementation.
Negative write slots, a live physical slot zero, prefix-cache restoration and
speculative acceptance need explicit adaptation before that framework switches
prep ownership. The prepared API remains available throughout that migration.

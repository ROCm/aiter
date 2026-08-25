# Qwen Sparse Attention

AITER provides portable Triton implementations of the Qwen-Air QSA path in
`aiter.ops.triton.attention.qsa`, plus validated gfx950 Gluon specializations
for paged MQA scoring and sparse paged GQA.

## Operators

- `qsa_paged_mqa_logits` reads BF16 compressed keys through a request page
  table and computes the QSA `sum(ReLU(q @ k)) / sqrt(head_dim)` score.
- `qsa_select_paged_tokens` combines the scoring kernel with AITER's radix
  `top_k_per_row_prefill` implementation and expands selected compressed groups
  to logical token indices. The current incomplete causal group is appended.
- `qsa_sparse_paged_gqa` consumes those logical token indices and applies
  online-softmax grouped-query attention directly over separate paged BF16
  K/V caches.

The cache layout is `[pages, page_size, kv_heads, head_dim]`. Compressed scoring
uses one KV head. Request page tables contain physical page IDs, while selected
indices remain logical token positions.

## vLLM integration

The AITER API matches the three stages in
`vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py`. A vLLM ROCm integration can
replace:

1. `qsa_mqa_paged` and the adjacent top-k/expansion calls with
   `qsa_select_paged_tokens`.
2. `qsa_sparse_paged_attention` with `qsa_sparse_paged_gqa`.

The portable Triton kernels remain available on every supported architecture
and are the fallback for unsupported Gluon shapes and failures.

## gfx950 Gluon dispatch

The public operators accept `backend=None`, `"auto"`, `"triton"`, or
`"gluon"`. `None` is equivalent to `"auto"`.

Paged MQA scoring auto-selects Gluon only when all of the following hold:

- the device architecture is gfx950 and the Gluon kernels imported
  successfully with Triton 3.6 or newer;
- the query is BF16 with shape `[tokens, 4, 128]` (the released Qwen3.8
  checkpoint) or `[tokens, 8, 128]` (the earlier Qwen-Air checkpoint);
- `compress_ratio` is 4; and
- the query, compressed K cache, and output can use signed 32-bit buffer
  offsets.

All other scoring cases use Triton. In `"auto"` mode, a Gluon JIT, compiler, or
launch failure is caught and retried with Triton. `"triton"` always selects the
portable kernel. `"gluon"` requires the exact supported shape and environment;
an unsupported configuration raises an error, and runtime failures are
reported instead of being hidden by fallback. `qsa_select_paged_tokens` passes
the same backend choice through to its scoring stage.

Sparse GQA auto-selects Gluon for the validated Qwen-Air geometry: gfx950,
Triton 3.6 or newer, BF16 queries and caches, head dimension 128, GQA group
size 5, selection width 2051, and signed 32-bit buffer offsets for the query,
K/V caches, and output. The specialization uses a 64-column/four-warp tile,
cache-all buffer loads, and separate handling for the three-entry tail after
the 2048 full columns. `"triton"` remains a portable override. Unsupported
automatic configurations and automatic Gluon failures fall back to Triton;
unsupported forced configurations and forced Gluon runtime failures raise an
error. Large-address cases therefore remain on Triton.

## Validation and performance

On gfx950, the QSA test suite passed, including forced Triton/Gluon parity for
page boundaries, causal masking, invalid indices, multiple request page
tables, head dimension 128, Qwen-Air grouped-query geometry, and selection
width 2051. The vLLM QSA integration tests passed (3 passed, 5 deselected), and
a TP=8 completion smoke test returned HTTP 200 while exercising both AITER QSA
integration paths.

Representative median kernel times were:

- released Qwen3.8 paged MQA scorer, query shape `(32, 4, 128)` with 4096
  columns: Triton 0.018383 ms, Gluon 0.017566 ms (4.5% lower latency, 1.047x);
- paged MQA scorer, query shape `(32, 8, 128)` with 4096 columns: Triton
  0.0229 ms, Gluon 0.0164 ms (28% faster);
- sparse GQA, query shape `(16, 10, 128)` with selection width 2051 and ordered
  indices: Triton 0.123241 ms, Gluon 0.103560 ms (16.0% lower latency, 1.190x);
- the same sparse GQA shape with randomized production-like indices: Triton
  0.120121 ms, Gluon 0.104321 ms (13.2% lower latency, 1.151x).

The sparse GQA figures are p50 results independently reproduced from clean
caches across seven alternating-order runs; p50 coefficient of variation was
0.015--0.025%.

Consequently, both validated gfx950 specializations auto-select Gluon.

## Current constraints

- BF16 query and cache tensors.
- Separate K and V caches.
- Integer page tables and request metadata.
- `token_topk` must be divisible by `compress_ratio`.
- AITER's compiled `module_top_k_per_row` extension is required for the fused
  selection pipeline.

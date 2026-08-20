# Qwen Sparse Attention

AITER provides a correctness-first Triton implementation of the Qwen-Air QSA
path in `aiter.ops.triton.attention.qsa`.

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

Keep the existing Triton implementation as a fallback until representative
prefill and decode shapes have passed performance gates. The first AITER
version is portable Triton; a gfx950 Gluon specialization is not included yet.

## Current constraints

- BF16 query and cache tensors.
- Separate K and V caches.
- Integer page tables and request metadata.
- `token_topk` must be divisible by `compress_ratio`.
- AITER's compiled `module_top_k_per_row` extension is required for the fused
  selection pipeline.

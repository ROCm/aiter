# Production scale — the facts a 32-bit overflow judgement needs

Whether an index x stride product can exceed 2^31 depends on deployment scale, which the
diff does not contain. Without these numbers a reviewer cannot name a triggering case and
correctly clears every candidate above. Snapshot -- keep sourced and current; a stale row
produces a confidently wrong verdict.

The first three rows are partly **in-sample**: they come from the problem statements of the
fix PRs that also supplied the known-defect labels used during development. They demonstrate
that scale facts can make the arithmetic decidable; they do not establish holdout recall or
generalize to deployments with different limits. Replace them with deployment-config sources
before treating this table as production policy.

| quantity | scale | source |
|---|---|---|
| DeepSeek-V4 unified KV pool | ~150M rows | aiter#4680 problem statement |
| Sparse-indexer decode batch | up to 512 concurrent sequences | aiter#4244 problem statement |
| KV stride passed by callers | may be a per-group page size, not a per-token stride, when the cache is a strided view into one shared allocation | aiter#4774 problem statement |
| Production token range | 1 -> 16384 per launch | P2, review-pr |
| MoE production configs | DSv4 E=385/topk=7; GPT-OSS 120B; Kimi-K2.5 | P2, review-pr |

Worked example: `stride = KVBlockSize * index_dim`; at `KVBlockSize=256, index_dim=132` a
block index of ~63.5K puts the product past 2^31 -- well inside a 150M-row pool.

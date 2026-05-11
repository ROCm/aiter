---
title: "Multi-head Latent Attention (MLA)"
last_verified: 2026-04-06
source_files:
  - aiter/mla/
tags: [mla, attention, deepseek, latent-attention]
---

# Multi-head Latent Attention (MLA)

## Overview
MLA (Multi-head Latent Attention) is a specialized attention variant used in models like DeepSeek-V2/V3. It compresses the KV cache by projecting keys and values into a lower-dimensional latent space, significantly reducing memory usage for long-context inference.

## Implementation
aiter provides MLA as a separate submodule (`aiter.mla`), imported directly in `aiter/__init__.py`. The MLA implementation uses a specialized KV cache layout compatible with FlashInfer's page-table layout.

## KV Cache Layout
MLA uses a different KV cache layout than standard MHA/GQA. The KV cache stores latent representations rather than full key/value tensors, with the latent dimension being much smaller than the original head dimension times number of KV heads.

## Related Pages
- [[operators/attention]] -- standard attention variants
- [[operators/cache]] -- KV cache management
- [[operators/rope]] -- position encoding in MLA context

## Source Files
- `aiter/mla/` -- MLA submodule

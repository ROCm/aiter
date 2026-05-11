---
title: "Sampling Operators"
last_verified: 2026-04-06
source_files:
  - aiter/ops/topk.py
  - aiter/ops/topk_plain.py
  - aiter/ops/sample.py
  - aiter/ops/sampling.py
tags: [sampling, topk, inference]
---

# Sampling Operators

## Overview
aiter provides top-k sampling operators for token generation during autoregressive inference.

## Key Functions
- `topk` -- top-k selection (CK-compiled)
- `topk_plain` -- plain top-k implementation
- `sample` -- sampling from distributions
- `sampling` -- sampling utilities

## Related Pages
- [[operators/moe]] -- MoE uses top-k for expert selection
- [[operators/attention]] -- attention output feeds into sampling

## Source Files
- `aiter/ops/topk.py` -- top-k operator
- `aiter/ops/topk_plain.py` -- plain top-k
- `aiter/ops/sample.py` -- sampling
- `aiter/ops/sampling.py` -- sampling utilities

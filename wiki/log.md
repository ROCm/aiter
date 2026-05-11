---
title: "Wiki Log"
tags: [log]
---

# Wiki Log

Chronological record of wiki operations.

---

## [2026-04-06] ingest | Initial Wiki Creation

Created the full wiki structure with 18 pages from codebase analysis.

**Pages created:**
- `overview.md` -- architecture overview from README.md, __init__.py, setup.py, docs/
- `operators/attention.md` -- from docs/attention_pipelines.md, docs/attention_kernels_explained.md, ops/attention.py, ops/mha.py, ops/unified_attention.py, ops/triton/attention/
- `operators/gemm.md` -- from ops/gemm_op_a8w8.py, ops/gemm_op_a16w16.py, ops/gemm_op_a4w4.py, ops/batched_gemm_*, ops/deepgemm.py, configs/
- `operators/moe.md` -- from ops/moe_op.py, ops/moe_sorting.py, ops/activation.py, configs/tuned_fmoe.csv
- `operators/quant.md` -- from ops/quant.py
- `operators/norm.md` -- from ops/norm.py, ops/rmsnorm.py, ops/groupnorm.py
- `operators/rope.md` -- from ops/rope.py, ops/pos_encoding.py, ops/fused_qk_norm_rope_cache_quant.py
- `operators/cache.md` -- from ops/cache.py
- `operators/communication.md` -- from ops/communication.py, ops/custom_all_reduce.py, docs/triton_comms.md
- `operators/sampling.md` -- from ops/topk.py, ops/sample.py
- `operators/mla.md` -- from aiter/mla/
- `concepts/backend-selection.md` -- from __init__.py, ops/triton/attention/unified_attention.py, ops/gemm_op_a8w8.py
- `concepts/autotuning.md` -- from docs/autotuning_pipeline.md, configs/
- `concepts/jit-compilation.md` -- from aiter/jit/core.py, jit/utils/chip_info.py
- `concepts/hardware-targets.md` -- from jit/utils/chip_info.py, 3rdparty/composable_kernel/README.md
- `ck/architecture.md` -- from 3rdparty/composable_kernel/README.md
- `ck/attention-pipeline.md` -- from 3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/
- `integration/vllm.md` -- from ../vllm-aiter-silu-mul/README.md

**Also created:**
- `AGENTS.md` -- wiki schema file with conventions, page format, and ingest/query/lint workflows
- `wiki/index.md` -- master catalog of all pages
- `wiki/log.md` -- this file

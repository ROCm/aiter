# Batched full-layer implementation

Target: Flash TP1 C2/C4/C8, same complete operation sequence as corrected C1. One256-CTA launch per layer. This directory is under development; no full-layer batched performance result yet.

| Stage | Assignment | Publication / dependency |
|---|---|---|
| mHC projection | 256 CTAs, shared BF16 WMMA weights, M valid rows | partials[M,128,24], resident-grid barrier |
| mHC RMS/mix/Sinkhorn | 2 CTAs/token, preserve C1 FP32 statistic order | disjoint mix[M,24], x[M,4096], barrier |
| Input quant / QA/KV / QB | existing batched scaled-WMMA region,256 CTAs | token-indexed partials, masked L2 prefetch, early TDM |
| Compressor projections | M rows in BF16 WMMA tiles | per-token plans, state slots, guarded cache writes |
| Attention | token/head/split tasks distributed over256 CTAs | separate softmax partials per token, grid barrier before reduction |
| Output projections | M rows in FP8 WMMA tiles, weights shared | partials[split,M,N], per-token reduction/quant CTAs |
| mHC post / FFN pre | token/tile CTAs and2 reduction CTAs/token | per-token FP32 residuals, barrier |
| Router/shared expert | AITER GEMM M argument, fixed physical WMMA tile | row masks and row-indexed scales |
| Routing/routed experts | native top6 per row, expert-grouped row tasks | mapping/gather/scatter sized M*6, separate stage counters |
| FFN sum/post | token/tile CTAs | disjoint final outputs and preserved FP32 residual |

Scratch is sized by M. Readiness counters are stage-scoped and shared only when all256 CTAs participate. Do not wrap the full C1 body in a token loop. Per-stage tile scheduling loops are permitted. Preserve baseline reduction order and quantization rounding; validate distinct rows, not repeated copies.

Sources studied: installed AITER gfx1250 gemm_mxfp8.py (real M, masked rows, row-indexed scales), quant/quant.py (row/block assignment), gemm_a16w16.py; existing adapted AITER sparse attention and routed MoE kernels. Each primitive is compared with C1 on independent token rows, then with native ATOM in serving. Resource target: zero spills and fully resident256-CTA grid; inspect compiled metadata before serving.

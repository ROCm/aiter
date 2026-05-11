---
title: "Wiki Index"
last_verified: 2026-04-06
tags: [index]
---

# aiter Wiki Index

Master catalog of all wiki pages. Read this first to find relevant pages.

---

## Overview

| Page | Summary | Tags |
|------|---------|------|
| [[overview]] | High-level architecture: layer stack, directory map, operator families, config system | architecture, overview |

## Operators

| Page | Summary | Tags |
|------|---------|------|
| [[operators/attention]] | 8 attention kernel variants (Triton 2D/3D, CK-UA/SK/PK/Fwd, PA Decode/Prefill), selector logic, feature matrix, GQA head-merging | attention, triton, ck, paged-kv, gqa, flash-attention |
| [[operators/gemm]] | GEMM families (A8W8, A16W16, A4W4, batched, DeepGEMM), tuned CSV configs, M-padding, backend selection | gemm, ck, asm, triton, quantization, fp8, int4 |
| [[operators/moe]] | MoE pipeline: gating/top-k, token sorting, expert GEMM (CK fused vs Triton fallback), activation, FlyDSL | moe, triton, ck, asm, experts, routing |
| [[operators/quant]] | Quantization: SmoothQuant, per-token FP8/INT8, FP4 (per-1x32 block), MXFP4, dynamic quant | quantization, fp8, fp4, int8, smoothquant |
| [[operators/norm]] | RMSNorm, LayerNorm, GroupNorm with fused variants (add, smoothquant, dynamic quant) | normalization, rmsnorm, layernorm, groupnorm, ck |
| [[operators/rope]] | Rotary Position Embedding: NEOX/GPT-J styles, fwd/bwd, fused with norm+cache+quant | rope, position-encoding, ck |
| [[operators/cache]] | KV cache: paged block management (swap, copy), reshape_and_cache with optional quant | cache, kv-cache, paged-attention |
| [[operators/communication]] | Distributed: custom all-reduce, quick all-reduce, Iris Triton comms (reduce-scatter, all-gather) | communication, allreduce, iris, triton, distributed |
| [[operators/sampling]] | Top-k selection and sampling for autoregressive token generation | sampling, topk, inference |
| [[operators/mla]] | Multi-head Latent Attention (DeepSeek-style) with compressed KV cache | mla, attention, deepseek, latent-attention |

## Concepts

| Page | Summary | Tags |
|------|---------|------|
| [[concepts/backend-selection]] | Decision logic for ASM vs CK vs Triton, per operator family, fallback behavior | backend, triton, ck, asm, selection |
| [[concepts/autotuning]] | CI pipeline for generating per-shape per-arch tuned CSV configs, config file format | autotuning, configs, performance, ci |
| [[concepts/jit-compilation]] | JIT compilation of HIP/CK extensions via compile_ops(), GPU detection, torch.compile compat | jit, compilation, hip, build |
| [[concepts/hardware-targets]] | AMD GPU families (gfx908/90a/942/1100+), CU counts, MFMA vs WMMA, runtime detection | hardware, gpu, gfx, rocm, mi300 |

## Composable Kernel

| Page | Summary | Tags |
|------|---------|------|
| [[ck/architecture]] | CK four-layer model (tile ops -> templated kernel -> instances -> client API), CK vs CK Tile | ck, composable-kernel, architecture, hip |
| [[ck/attention-pipeline]] | CK Unified Attention Pipeline: template structure, tile tiers, GQA head-merge, MFMA flow | ck, attention, unified-attention, pipeline, mfma |

## Integration

| Page | Summary | Tags |
|------|---------|------|
| [[integration/vllm]] | vLLM integration: attention dispatch, MoE activation replacement, VLLM_ROCM_USE_AITER gating | vllm, integration, inference, rocm |

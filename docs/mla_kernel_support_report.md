# MLA Kernel Support Report: Triton vs ASM Backend Comparison

## Overview

This report provides a comprehensive analysis of Multi-head Latent Attention (MLA) kernel support in AITER, comparing the **Triton** and **ASM (Assembly)** backends across precision, fusion depth, execution modes, and feature coverage.

MLA is a key attention mechanism used in DeepSeek-style models. AITER provides two primary backends for MLA:

- **ASM**: Hand-tuned assembly kernels with pre-compiled binaries per GPU architecture. Production-grade with broad mode coverage.
- **Triton**: Python-based GPU kernels offering flexibility, portability, and rapid development. Growing coverage with unique features like sparse attention and FP4 GEMM fusion.

---

## 1. Precision / Data Type Support

| Data Type | Triton Decode | Triton Sparse | Triton Fused BMM+Cache | ASM Decode | ASM Prefill |
|-----------|:---:|:---:|:---:|:---:|:---:|
| BF16 Q x BF16 KV | Yes | Yes | Yes (activation) | Yes | Yes |
| FP16 | Yes (implicit) | - | - | - | - |
| FP32 accumulator | Yes | Yes | - | - | - |
| FP8 Q x FP8 KV | - | - | Yes (GEMM weights) | Yes (gqa=16,128) | Yes (gqa=1,128) |
| BF16 Q x FP8 KV | - | - | - | Yes (gqa=16, ps=1) | - |
| FP4 (MXFP4) | - | - | Yes (GEMM weights) | - | - |
| BF16 Q x byte KV | - | - | - | Yes (page64_ds32) | - |

**Key Findings:**

- ASM has broader native precision coverage for attention computation (BF16xFP8, FP8xFP8).
- Triton's FP8/FP4 support lives in the fused BMM+RoPE+Cache kernel (pre-attention GEMM), not in the attention kernel itself.
- Neither backend supports INT8 for MLA attention.

---

## 2. Fusion Levels

| Fused Operation | Triton Decode | Triton Sparse | Triton Fused Cache | ASM Decode |
|----------------|:---:|:---:|:---:|:---:|
| QK dot product | Yes (stage 1) | Yes | - | Yes (native) |
| Softmax | Yes (stage 1) | Yes | - | Yes (native) |
| V reduction | Yes (stage 1) | Yes | - | Yes (native) |
| RoPE on Q,K | Yes (in stage 1) | Yes (in kernel) | Yes (separate kernel) | Yes (native) |
| KV cache write | Separate | Separate | Yes (fused) | Separate |
| BMM (latent projection) | - | - | Yes (FP4/FP8 GEMM) | - |
| Quant scale application | - | - | Yes (k_scale, v_scale) | - |
| Split-K reduce | Stage 2 (Triton) | N/A (no split-K) | Split-K for FP4 BMM | Stage 2 (Triton) |
| Logit capping | Yes | - | - | No (must be 0) |

**Key Findings:**

- ASM fuses QK+softmax+V+RoPE into a single hand-tuned kernel with minimal overhead.
- Triton achieves similar fusion but through a 2-kernel pipeline (stage 1 + stage 2).
- Triton's unique advantage is the fused BMM+RoPE+Cache kernel that combines the latent projection GEMM (FP4/FP8) with RoPE and cache write in one pass -- something ASM does not have.

---

## 3. Execution Modes

| Mode | Triton | ASM | Notes |
|------|:---:|:---:|-------|
| Decode (seqlen_q=1) | Yes | Yes (22+ kernel variants) | Both well-supported |
| Decode (seqlen_q=2,4,8) | - | Yes (gqa=16) | ASM only for multi-token decode |
| Prefill (causal) | - | Yes (bf16, fp8) | ASM only |
| Prefill (non-causal) | - | Yes (fp8, gqa=1, ps=1) | ASM only |
| Persistent scheduling | - | Yes | ASM only -- constant GPU occupancy |
| Sparse (Top-K) | Yes | - | Triton only |

**Key Findings:**

- ASM covers decode + prefill + persistent. Triton decode is limited to seqlen_q=1 and lacks prefill.
- Triton's sparse MLA (Top-K attention) is unique but noted as "not optimized" in the source code.

---

## 4. GQA (Grouped Query Attention) Ratio Support

| GQA Ratio | Triton Decode | ASM |
|-----------|:---:|:---:|
| 1 (full MHA) | Yes | Yes (fp8 prefill only) |
| 16 | Yes | Yes (most kernels) |
| 32 | Yes | Yes (bf16 decode) |
| 64 | Yes | Yes (bf16 decode) |
| 128 | Yes | Yes (bf16+fp8 decode) |
| Arbitrary | Yes (`kv_group_num` param) | No (hardcoded per kernel) |

**Key Findings:**

- Triton accepts any GQA ratio via a runtime parameter.
- ASM requires a pre-compiled kernel binary per GQA ratio. Only ratios 1, 16, 32, 64, and 128 are currently available.

---

## 5. KV Cache Layout Support

| Feature | Triton Decode | Triton Fused Cache | ASM |
|---------|:---:|:---:|:---:|
| Paged cache | Yes | Yes | Yes |
| Page size = 1 | Yes (only) | Configurable | Yes |
| Page size > 1 | No | Yes | Yes (4, 64, etc.) |
| Flash layout `(T, blk, KH, D)` | - | Yes | - |
| Non-flash layout `(T, KH, D//x, blk, x)` | - | Yes | - |
| Quantized cache (k_scale, v_scale) | - | Yes | FP8 in some modes |
| 3-buffer split (nope + scale + rope) | - | - | Yes (persistent mode) |
| Partial last pages | Yes | Yes | Yes |

**Key Findings:**

- Triton decode is restricted to page_size=1. ASM supports multiple page sizes.
- The fused cache Triton kernels support the most layout flexibility (flash + non-flash).
- ASM persistent mode uses a unique 3-buffer split KV cache layout.

---

## 6. Split-K / Parallelism Strategy

| Aspect | Triton Decode | ASM Decode |
|--------|---------------|------------|
| Architecture | Stage 1 (Triton) + Stage 2 (Triton) | Stage 1 (ASM) + Stage 2 (Triton) |
| Split count | 1-16, tunable | 1-16, tunable |
| Auto-tuning | Yes (`_get_config` per arch) | Yes (metadata-based CU scheduling) |
| Reduce kernel | Triton `_fwd_kernel_stage2` | Triton `_fwd_kernel_stage2_asm` |
| Metadata | Simple index math | Dedicated CUDA kernel (v1.0/v1.1/v1.2) |

**Key Findings:**

- Both use split-K with Triton reduction. Even the ASM path depends on Triton for its stage 2 reduce kernel.
- ASM has a more sophisticated metadata system with dedicated GPU kernels for work distribution across CUs.

---

## 7. RoPE Handling

| Feature | Triton | ASM |
|---------|:---:|:---:|
| NeoX-style (interleaved) | Yes | Yes (native) |
| GPT-J style (block) | Yes | Yes (native) |
| Partial rotary dim | Yes | Yes |
| RoPE offset | No (TODO in code) | Yes |
| Cosine embedding | Yes (fused cache variant) | - |

---

## 8. Hardware Coverage

| GPU | Triton | ASM |
|-----|:---:|:---:|
| GFX942 (MI300X) | Yes (tuned configs) | Yes (22 `.co` binaries) |
| GFX950 (MI350) | Yes (tuned configs) | Yes (22 `.co` binaries) |
| Other GPUs | Portable (any Triton target) | No (arch-specific ASM) |

---

## 9. Test and Benchmark Coverage

### Triton Tests

| Test File | Coverage |
|-----------|----------|
| `op_tests/triton_tests/attention/test_mla_decode_rope.py` | RoPE decode, split-K, paged cache |
| `op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py` | Top-K sparse, block tables |
| `op_tests/triton_tests/utils/mla_decode_ref.py` | Reference decode implementation |
| `op_tests/triton_tests/utils/mla_extend_ref.py` | Reference extend implementation |

### ASM Tests

| Test File | Coverage |
|-----------|----------|
| `op_tests/test_mla.py` | Decode, variable context, dtype combos, page sizes |
| `op_tests/test_mla_persistent.py` | Persistent mode, FP8, 3-buffer cache |
| `op_tests/test_mla_prefill_ps.py` | Prefill with persistent scheduling |
| `op_tests/test_mla_sparse.py` | Sparse patterns, TopK, FP8 |
| `op_tests/test_concat_cache_mla.py` | Cache concatenation ops |

### Benchmarks

| Benchmark File | Coverage |
|----------------|----------|
| `op_tests/op_benchmarks/triton/bench_mla_decode.py` | Decode performance |
| `op_tests/op_benchmarks/triton/bench_mla_decode_rope.py` | RoPE decode performance |

---

## 10. Summary: Backend Strengths

| Dimension | Triton | ASM | Advantage |
|-----------|--------|-----|-----------|
| Data type coverage | BF16, FP16, FP32; FP4/FP8 in fused BMM | BF16, FP8 native attention | Triton (more general) |
| Fusion depth | Medium (RoPE+attention or BMM+RoPE+cache) | High (all-in-one native kernel) | ASM |
| Execution modes | Decode only | Decode, prefill, persistent | ASM |
| GQA flexibility | Any ratio | 1, 16, 32, 64, 128 | Triton |
| Sparse support | Top-K sparse MLA | None | Triton |
| Performance potential | Good (compiler-optimized) | Excellent (hand-tuned ASM) | ASM |
| Development velocity | High (Python-based) | Low (assembly) | Triton |
| Production maturity | Growing | Very high (22+ kernel variants) | ASM |
| Portability | Any Triton-supported GPU | GFX942 and GFX950 only | Triton |
| Cache layout flexibility | Flash + non-flash (fused cache) | 3-buffer split, multi-page | Comparable |

---

## Conclusion

The ASM and Triton backends are **complementary**:

- **ASM** is the production workhorse, covering decode, prefill, and persistent modes with hand-tuned performance on MI300X and MI350 GPUs. It is the recommended choice for latency-critical inference workloads.
- **Triton** fills important gaps: sparse attention (Top-K), FP4 GEMM fusion, arbitrary GQA ratios, and GPU portability. It also serves as the development platform for rapid prototyping of new MLA features.
- Both backends share the Triton-based split-K reduction kernel, demonstrating a practical hybrid approach.

### Recommended Areas for Future Development

1. **Triton prefill support**: Adding prefill mode to the Triton MLA decode kernel would close the largest feature gap.
2. **RoPE offset in Triton**: Currently marked as TODO in the codebase.
3. **Triton page_size > 1**: The decode kernel is limited to page_size=1; supporting larger pages would improve memory efficiency.
4. **ASM GQA flexibility**: Supporting additional GQA ratios beyond the current set (1, 16, 32, 64, 128).
5. **Triton sparse MLA optimization**: The current implementation is noted as "not optimized" in the source.

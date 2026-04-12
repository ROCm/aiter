Attention Operations
====================

AITER provides GPU-optimized attention kernels for both training (MHA with forward
and backward passes) and inference (Paged Attention, Multi-Latent Attention).
All kernels target AMD Instinct GPUs via ROCm.

.. contents:: Sections
   :local:
   :depth: 1


Multi-Head Attention (Flash)
----------------------------

Flash attention implementations with CK (Composable Kernel) backends.
Used for both training and inference. Located in ``aiter.ops.mha``.

High-Level API
~~~~~~~~~~~~~~

These are the primary user-facing functions with ``torch.autograd`` support.


``flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1, 0), bias=None, alibi_slopes=None, deterministic=True, return_lse=False, return_attn_probs=False, how_v3_bf16_cvt=1, cu_seqlens_q=None, cu_seqlens_kv=None, sink_ptr=None)``

Standard flash attention forward/backward with autograd.
Supports MQA/GQA via fewer KV heads, causal masking, sliding window, ALiBi slopes,
and FP8 inputs. Dispatches to CK or FMHA v3 backend based on dtype and arch.

- **q**: ``(batch, seqlen, nheads, headdim_q)``
- **k**: ``(batch, seqlen, nheads_k, headdim_q)``
- **v**: ``(batch, seqlen, nheads_k, headdim_v)``
- **Returns**: ``out (batch, seqlen, nheads, headdim_v)``, optionally ``softmax_lse``, ``S_dmask``


``flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, min_seqlen_q=0, dropout_p=0.0, softmax_scale=None, logits_soft_cap=0.0, causal=False, window_size=(-1, -1, 0), bias=None, alibi_slopes=None, deterministic=False, return_lse=False, return_attn_probs=False, how_v3_bf16_cvt=1, block_table=None, out=None, ...)``

Variable-length flash attention with autograd. Sequences are packed into a single
tensor and indexed by cumulative sequence lengths.

- **q**: ``(total_q, nheads, headdim_q)``
- **k**: ``(total_k, nheads_k, headdim_q)``
- **v**: ``(total_k, nheads_k, headdim_v)``
- **cu_seqlens_q**: ``(batch_size + 1,)`` cumulative query lengths
- **cu_seqlens_k**: ``(batch_size + 1,)`` cumulative key lengths
- **Returns**: ``out (total_q, nheads, headdim_v)``, optionally ``softmax_lse``, ``S_dmask``

FP8 Convenience Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~


``flash_attn_fp8_pertensor_func(q, k, v, q_descale, k_descale, v_descale, causal=False, window_size=(-1, -1, 0), softmax_scale=None, sink_ptr=None)``

Flash attention for FP8 inputs with per-tensor descaling. Forward-only (no autograd).


``flash_attn_varlen_fp8_pertensor_func(q, k, v, q_descale, k_descale, v_descale, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)``

Variable-length FP8 flash attention with per-tensor descaling. Forward-only.

Batch Prefill
~~~~~~~~~~~~~


``mha_batch_prefill_func(q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, logits_soft_cap=0.0, causal=False, window_size=(-1, -1), alibi_slopes=None, ...)``

Paged KV cache batch prefill attention. Supports both vectorized (5D) and linear (3D/4D)
KV cache layouts.

Low-Level CK Kernels
~~~~~~~~~~~~~~~~~~~~~

These are the direct CK kernel wrappers. Most users should prefer the high-level API above.


``mha_fwd(q, k, v, dropout_p, softmax_scale, is_causal, window_size_left, window_size_right, sink_size, return_softmax_lse, return_dropout_randval, ...)``

CK flash attention forward pass. Returns ``(out, softmax_lse, S_dmask, rng_state)``.


``fmha_v3_fwd(q, k, v, dropout_p, softmax_scale, is_causal, window_size_left, window_size_right, return_softmax_lse, return_dropout_randval, how_v3_bf16_cvt, ...)``

FMHA v3 forward pass (newer CK backend). Returns ``(out, softmax_lse, S_dmask, rng_state)``.


``mha_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, min_seqlen_q, dropout_p, softmax_scale, logits_soft_cap, zero_tensors, is_causal, window_size_left, window_size_right, sink_size, return_softmax_lse, return_dropout_randval, ...)``

Variable-length CK MHA forward. Returns ``(out, softmax_lse, S_dmask, rng_state)``.


``fmha_v3_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, min_seqlen_q, dropout_p, softmax_scale, logits_soft_cap, zero_tensors, is_causal, window_size_left, window_size_right, return_softmax_lse, return_dropout_randval, how_v3_bf16_cvt, ...)``

FMHA v3 variable-length forward pass.


``mha_bwd(dout, q, k, v, out, softmax_lse, dropout_p, softmax_scale, is_causal, window_size_left, window_size_right, deterministic, dq=None, dk=None, dv=None, ...)``

CK MHA backward pass (training). Returns ``(dq, dk, dv, dbias)``.


FMHA v3 backward pass (training).


Variable-length CK MHA backward pass (training).


FMHA v3 variable-length backward pass (training).


Paged Attention
---------------

Paged attention kernels for LLM decode-phase inference with block-based KV caches.
Located in ``aiter.ops.attention``.

Core Functions
~~~~~~~~~~~~~~


``paged_attention_rocm(out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, fp8_out_scale=None, partition_size=256, mtp=1, q_scale=None)``

Main ROCm paged attention entry point. Custom CK-based implementation with
partitioned softmax. Supports FP8 KV cache, ALiBi, and multi-token prediction (MTP).


``paged_attention_v1(out, workspace_buffer, query, key_cache, value_cache, scale, block_tables, cu_query_lens, context_lens, max_context_len, alibi_slopes, kv_cache_dtype, kv_cache_layout, logits_soft_cap, k_scale, v_scale, fp8_out_scale=None, partition_size=256, mtp=1, sliding_window=0)``

V1 paged attention with workspace buffer. Supports multiple KV cache layouts,
logits soft capping, and sliding window attention.


``paged_attention_ragged(out, workspace_buffer, query, key_cache, value_cache, scale, kv_indptr, kv_page_indices, kv_last_page_lens, block_size, max_num_partitions, alibi_slopes, kv_cache_dtype, kv_cache_layout, logits_soft_cap, k_scale, v_scale, fp8_out_scale=None, partition_size=256, mtp=1)``

Ragged tensor paged attention. Uses indirect page indexing (``kv_indptr``,
``kv_page_indices``) instead of dense block tables.

ASM Paged Attention
~~~~~~~~~~~~~~~~~~~

Hand-tuned assembly kernels for maximum decode throughput.


``pa_fwd_asm(Q, K, V, block_tables, context_lens, block_tables_stride0, max_qlen=1, K_QScale=None, V_QScale=None, out_=None, qo_indptr=None, high_precision=1, kernelName=None)``

ASM paged attention forward. Supports FP8 KV cache via dequantization scales
(``K_QScale``, ``V_QScale``). The ``high_precision`` parameter controls FP8
accumulation precision (0=low, 1=medium, 2=highest).


``pa_ps_fwd_asm(Q, K, V, kv_indptr, kv_page_indices, context_lens, softmax_scale, max_qlen=1, K_QScale=None, V_QScale=None, out_=None, qo_indptr=None, work_indptr=None, work_info=None, splitData=None, splitLse=None, mask=0, high_precision=1, kernelName=None, quant_type=QuantType.per_Token)``

PS-mode (persistent/split) ASM paged attention. Uses ragged page indexing and
supports work partitioning for large context lengths.


``pa_persistent_fwd(Q, K, V, output, max_qlen, qo_indptr, kv_indptr, kv_indices, context_lens, work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map, K_QScale=None, V_QScale=None, softmax_scale=None, mask=0, quant_type=QuantType.per_Token)``

Persistent paged attention combining PS-mode forward with reduction.
Orchestrates ``pa_ps_fwd_asm`` + ``pa_reduce_v1`` for long-context decode.
Returns ``(logits, final_lse)``.

vLLM-Compatible Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~

Drop-in replacement for vLLM's paged attention layer. Located in ``aiter.paged_attn``.

   :members: get_supported_head_sizes, get_kv_cache_shape, split_kv_cache, write_to_paged_cache, forward_decode, swap_blocks, copy_blocks

   :members:


``paged_attention_v1(out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, ...)``

vLLM-compatible v1 paged attention (delegates to ``aiter.ops``).


``paged_attention_v2(out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, ...)``

vLLM-compatible v2 paged attention with partitioned softmax.


Multi-Latent Attention (MLA)
----------------------------

Attention kernels for DeepSeek-style Multi-Latent Attention, where key and value
are projected into a shared low-rank latent space. Located in ``aiter.mla``.
All MLA functions are inference-only.


``mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, max_seqlen_q, page_size=1, nhead_kv=1, sm_scale=None, logit_cap=0.0, num_kv_splits=None, ...)``

MLA decode-phase forward pass. Operates on paged KV buffers with the latent
dimension fused into ``kv_buffer``. Supports both ASM and Triton backends
with automatic split/reduce for long contexts.

- **q**: ``(total_q, nheads, qk_head_dim)``
- **kv_buffer**: ``(num_pages, page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim)``
- **o**: ``(total_q, nheads, v_head_dim)`` output buffer


``mla_prefill_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, max_seqlen_q, sm_scale=None, logit_cap=0.0, num_kv_splits=None)``

MLA prefill-phase forward pass. Uses ASM backend for the attention computation.

- **q**: ``(num_seqs, num_heads, head_size)``
- **kv_buffer**: ``(num_pages, page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim)``
- **o**: ``(num_seqs, num_heads, v_head_dim)``


``mla_prefill_ps_fwd(Q, K, V, output, qo_indptr, kv_indptr, kv_page_indices, work_indptr, work_info_set, max_seqlen_q, is_causal, reduce_indptr=None, reduce_final_map=None, reduce_partial_map=None, softmax_scale=None, q_scale=None, k_scale=None, v_scale=None)``

MLA prefill with persistent/split mode. Handles long prefill sequences via
work partitioning and multi-stage reduction. Supports FP8 via per-tensor scales.


``mla_prefill_reduce(partial_output, partial_lse, reduce_indptr, reduce_final_map, reduce_partial_map, output, tile_q=256, use_triton=True)``

Reduction kernel for MLA prefill split outputs. Combines partial attention
outputs using log-sum-exp for numerically stable merging. Implemented in Triton
with a PyTorch fallback.

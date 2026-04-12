Core Operators
==============

AITER provides fused, high-performance operators for LLM inference on AMD GPUs.
All operators are JIT-compiled via the ``@compile_ops`` decorator from ``aiter.jit.core``
and target AMD Instinct GPUs (gfx942, gfx950, gfx1250).

Normalization
-------------

**Module:** ``aiter.ops.norm``, ``aiter.ops.rmsnorm``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``layer_norm(input, weight, bias, epsilon, x_bias)``
     - Layer normalization (CK-based)
   * - ``layernorm2d_fwd(input, weight, bias, epsilon, x_bias)``
     - 2D layer norm forward
   * - ``layernorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, bias, epsilon)``
     - Fused residual add + layer norm
   * - ``layernorm2d_fwd_with_smoothquant(out, input, xscale, yscale, weight, bias, epsilon)``
     - Fused layer norm + smooth quantization
   * - ``rms_norm(input, weight, epsilon)``
     - RMS normalization (returns normalized tensor)
   * - ``rmsnorm2d_fwd(input, weight, epsilon)``
     - 2D RMS norm forward (auto-selects CK or HIP backend)
   * - ``rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)``
     - Fused residual add + RMS norm
   * - ``rmsnorm2d_fwd_with_smoothquant(out, input, xscale, yscale, weight, epsilon)``
     - Fused RMS norm + smooth quantization
   * - ``rmsnorm2d_fwd_with_dynamicquant(out, input, yscale, weight, epsilon, ...)``
     - Fused RMS norm + dynamic quantization
   * - ``rmsnorm2d_fwd_with_add_dynamicquant(out, input, residual_in, residual_out, yscale, weight, epsilon, ...)``
     - Fused residual add + RMS norm + dynamic quantization
   * - ``add_rmsnorm_quant(out, input, residual_in, residual_out, scale, weight, epsilon, ...)``
     - Fused add + RMS norm + quantization
   * - ``rmsnorm_quant(out, input, scale, weight, epsilon, ...)``
     - RMS norm + quantization (no residual)
   * - ``add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon)``
     - Fused add + RMS norm (no quantization)

Activation
----------

**Module:** ``aiter.ops.activation``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``silu_and_mul(out, input)``
     - Fused SiLU activation + element-wise multiply
   * - ``scaled_silu_and_mul(out, input, scale)``
     - Scaled SiLU + multiply (for FP8 output)
   * - ``gelu_and_mul(out, input)``
     - Fused GELU + multiply
   * - ``gelu_tanh_and_mul(out, input)``
     - Fused GELU-tanh approximation + multiply
   * - ``gelu_fast(out, input)``
     - Fast GELU approximation

All activation ops write results in-place to the ``out`` tensor.

RoPE (Rotary Position Embedding)
---------------------------------

**Module:** ``aiter.ops.rope``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``rope_fwd_impl(output, input, freqs, rotate_style, reuse_freqs_front_part, nope_first)``
     - RoPE forward (single input, uncached)
   * - ``rope_bwd_impl(input_grads, output_grads, freqs, rotate_style, ...)``
     - RoPE backward
   * - ``rope_2c_fwd_impl(output_x, output_y, input_x, input_y, freqs, ...)``
     - RoPE forward for two inputs (Q + K)
   * - ``rope_2c_bwd_impl(...)``
     - RoPE backward for two inputs

Inputs use ``sbhd`` layout. ``rotate_style``: 0 = NeoX (rotate 2nd half), 1 = GPT-J (rotate odd elements).

Quantization
------------

**Module:** ``aiter.ops.quant``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``smoothquant_fwd(out, input, x_scale, y_scale)``
     - Smooth quantization forward
   * - ``moe_smoothquant_fwd(out, input, x_scale, topk_ids, y_scale)``
     - MoE-aware smooth quantization
   * - ``pertoken_quant(x, scale, x_scale, scale_dtype, quant_dtype)``
     - Per-token quantization (pure PyTorch)
   * - ``per_1x32_f4_quant(x, scale, quant_dtype)``
     - FP4 block quantization (1x32 group size)

**Triton quantization kernels** (``aiter.ops.triton.quant``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``static_per_tensor_quant_fp8_i8(qx, x_in, scale_in)``
     - Static per-tensor FP8/INT8 quantization
   * - ``dynamic_per_tensor_quant_fp8_i8(qx, x_in, scale_out)``
     - Dynamic per-tensor FP8/INT8 quantization
   * - ``dynamic_per_token_quant_fp8_i8(qx, x_in, scale_out)``
     - Dynamic per-token FP8/INT8 quantization
   * - ``dynamic_mxfp4_quant(...)``
     - Dynamic MXFP4 quantization

Sampling
--------

**Module:** ``aiter.ops.sample``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``greedy_sample(out, input)``
     - Greedy (argmax) sampling
   * - ``random_sample(out, input, temperatures, lambd, generator, eps)``
     - Random sampling with temperature
   * - ``random_sample_outer_exponential(out, input, exponentials, temperatures, eps)``
     - Random sampling with externally generated exponentials
   * - ``mixed_sample(out, input, temperature, lambd, generator, eps)``
     - Mixed greedy/random sampling (per-token temperature)
   * - ``mixed_sample_outer_exponential(out, input, exponentials, temperature, eps)``
     - Mixed sampling with external exponentials
   * - ``exponential(out, lambd, generator, eps)``
     - Generate exponential random variates

KV Cache
--------

**Module:** ``aiter.ops.cache``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, ...)``
     - Standard KV cache update
   * - ``reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)``
     - Flash-attention-style cache update
   * - ``reshape_and_cache_with_pertoken_quant(key, value, key_cache, value_cache, k_dequant_scales, v_dequant_scales, slot_mapping, asm_layout)``
     - Cache update with per-token FP8 quantization
   * - ``reshape_and_cache_with_block_quant(key, value, key_cache, value_cache, k_dequant_scales, v_dequant_scales, slot_mapping, asm_layout)``
     - Cache update with block-level quantization
   * - ``concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)``
     - Multi-Latent Attention (MLA) cache update
   * - ``copy_blocks(key_caches, value_caches, block_mapping)``
     - Copy cache blocks (for beam search)
   * - ``swap_blocks(src, dst, block_mapping)``
     - Swap cache blocks between devices

Top-K / MoE Gating
-------------------

**Module:** ``aiter.ops.topk``, ``aiter.ops.moe_op``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, need_renorm, ...)``
     - Fused top-k + softmax for MoE gating
   * - ``topk_sigmoid(topk_weights, topk_indices, gating_output)``
     - Fused top-k + sigmoid for MoE gating
   * - ``grouped_topk(gating_output, topk_weights, topk_ids, num_expert_group, topk_group, need_renorm, ...)``
     - Grouped top-k selection (e.g., DeepSeek MoE)
   * - ``biased_grouped_topk(gating_output, correction_bias, topk_weights, topk_ids, num_expert_group, topk_group, need_renorm, ...)``
     - Biased grouped top-k (auto-selects HIP or fused gate backend)
   * - ``moe_fused_gate(input, bias, topk_weights, topk_ids, num_expert_group, topk_group, topk, n_share_experts_fusion, ...)``
     - Fused MoE gating kernel
   * - ``moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)``
     - Align MoE token assignments to block boundaries
   * - ``moe_sum(input, output)``
     - Sum MoE expert outputs

Communication
-------------

**Module:** ``aiter.ops.custom_all_reduce``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``all_reduce(_fa, inp, out, use_new, open_fp8_quant, reg_inp_ptr, reg_inp_bytes)``
     - Custom all-reduce (P2P IPC-based)
   * - ``reduce_scatter(_fa, inp, out, reg_ptr, reg_bytes)``
     - Reduce-scatter
   * - ``fused_allreduce_rmsnorm(_fa, inp, res_inp, res_out, out, w, eps, reg_ptr, reg_bytes, use_1stage)``
     - Fused all-reduce + RMS normalization
   * - ``fused_allreduce_rmsnorm_quant(_fa, inp, res_inp, res_out, out, scale_out, w, eps, reg_ptr, reg_bytes, use_1stage)``
     - Fused all-reduce + RMS norm + quantization

See Also
--------

* :doc:`gemm` - Matrix multiplication operations (GEMM, batched GEMM, MoE GEMM)
* :doc:`attention` - Attention operations (paged attention, flash attention, MLA)
* :doc:`../tutorials/add_new_op` - How to add a new operator

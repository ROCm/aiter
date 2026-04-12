Mixture of Experts (MoE) API
=============================

AITER provides fused MoE kernels optimized for AMD GPUs. These are used by
inference engines (ATOM, vLLM, SGLang) for MoE model architectures such as
DeepSeek-V3/R1, Kimi K2.5, and GPT-OSS 120B.

Gating
------

.. py:function:: topk_softmax(topk_weights, topk_ids, token_expert_indicies, gating_output, topk)

   Fused top-k selection and softmax for MoE gating. Computes the top-k expert
   scores from ``gating_output`` and returns normalized weights.

   :param topk_weights: Output tensor for the normalized top-k weights.
   :param topk_ids: Output tensor for the selected expert indices.
   :param token_expert_indicies: Output tensor for token-to-expert mapping.
   :param gating_output: Raw gating logits of shape ``(num_tokens, num_experts)``.
   :param topk: Number of experts to select per token.

.. py:function:: topk_sigmoid(topk_weights, topk_ids, token_expert_indicies, gating_output, topk)

   Fused top-k selection and sigmoid gating. Similar to :func:`topk_softmax`
   but uses sigmoid activation instead of softmax.

Fused MoE
----------

.. py:function:: fmoe(input, w1, w2, topk_weights, topk_ids, ...)

   Main fused MoE forward pass. Dispatches tokens to selected experts, applies
   expert weights (w1, w2), and combines results.

   :param input: Hidden states of shape ``(num_tokens, hidden_dim)``.
   :param w1: First expert weight matrix.
   :param w2: Second expert weight matrix.
   :param topk_weights: Per-token expert weights from gating.
   :param topk_ids: Per-token expert indices from gating.

.. py:function:: fmoe_g1u1(input, gate, up, down, topk_weights, topk_ids, ...)

   Fused MoE with separate gate, up, and down projections (GLU-style).
   Used by architectures that split the MoE FFN into gate/up/down matrices.

.. py:function:: fmoe_int8_g1u0(...)

   INT8 quantized fused MoE. Applies INT8 weight-only quantization to the
   expert computations.

.. py:function:: fmoe_fp8_blockscale_g1u1(...)

   FP8 block-scale quantized fused MoE with gate/up/down projections.
   Uses per-block scaling factors for FP8 computation.

.. py:function:: fused_moe(hidden_states, w1, w2, gating_output, topk, ...)

   High-level fused MoE entry point (from ``aiter/fused_moe.py``). Combines
   gating and expert computation in a single call.

   :param hidden_states: Input hidden states.
   :param w1: First expert weight.
   :param w2: Second expert weight.
   :param gating_output: Raw gating logits.
   :param topk: Number of experts per token.

.. py:function:: fused_moe_2stages(...)

   Two-stage fused MoE: first sorts tokens by expert assignment, then runs
   expert computation. Can improve memory locality for large expert counts.

MoE Utilities
-------------

.. py:function:: moe_align_block_size(topk_ids, block_size, num_experts)

   Align MoE dispatch block sizes to hardware-friendly boundaries.

   :param topk_ids: Expert assignment indices.
   :param block_size: Target block size for alignment.
   :param num_experts: Total number of experts.

.. py:function:: moe_sorting(...)

   Sort tokens by their assigned expert. Used as a preprocessing step in
   two-stage MoE execution.

Source Files
------------

- ``aiter/ops/moe_op.py`` -- low-level MoE operations
- ``aiter/fused_moe.py`` -- high-level fused MoE interface

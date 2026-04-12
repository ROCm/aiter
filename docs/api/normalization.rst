Normalization API
=================

AITER provides optimized normalization kernels for AMD GPUs, including fused
variants that combine normalization with residual addition, quantization, or
both.

LayerNorm
---------

.. py:function:: layer_norm(input, weight, bias, eps)

   Standard layer normalization.

   :param input: Input tensor.
   :param weight: Learnable scale parameter.
   :param bias: Learnable bias parameter.
   :param eps: Small constant for numerical stability.
   :returns: Normalized tensor.

RMSNorm
-------

.. py:function:: rms_norm(input, weight, eps)

   Root Mean Square normalization. Simpler and faster than LayerNorm as it
   does not compute mean or use a bias term.

   :param input: Input tensor.
   :param weight: Learnable scale parameter.
   :param eps: Small constant for numerical stability.

.. py:function:: rmsnorm2d_fwd(input, weight, eps)

   2D RMS normalization forward pass. Operates on 2D input tensors of shape
   ``(batch, hidden_dim)``.

Fused Variants
--------------

These fused kernels combine RMSNorm with other operations to reduce memory
traffic and improve throughput.

.. py:function:: rmsnorm2d_fwd_with_add(input, residual, weight, eps)

   Fused residual addition and RMS normalization. Computes
   ``rmsnorm(input + residual)`` in a single kernel.

   :param input: Input tensor.
   :param residual: Residual tensor to add before normalization.
   :param weight: Learnable scale parameter.
   :param eps: Numerical stability constant.

.. py:function:: rmsnorm2d_fwd_with_smoothquant(...)

   Fused RMS normalization with SmoothQuant. Applies per-channel smooth
   quantization scales after normalization.

.. py:function:: rmsnorm2d_fwd_with_dynamicquant(...)

   Fused RMS normalization with dynamic quantization. Computes quantization
   parameters on the fly and outputs quantized activations.

.. py:function:: add_rmsnorm_quant(...)

   Fused residual add + RMS normalization + quantization in a single kernel.
   Combines three operations to minimize global memory round-trips.

Source Files
------------

- ``aiter/ops/norm.py`` -- LayerNorm and general norm operations
- ``aiter/ops/rmsnorm.py`` -- RMSNorm and fused RMSNorm variants

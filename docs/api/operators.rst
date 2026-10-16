Core Operators
==============

Normalization is documented in :doc:`normalization`. Many other AITER
operators use caller-owned output buffers. Check the source signature and
matching operator test instead of assuming an allocating PyTorch-style API.

Gated activations
-----------------

.. aiter-function:: aiter.ops.activation.silu_and_mul

The last input dimension holds concatenated gate and up values:
``input: (..., 2 * hidden)`` and ``out: (..., hidden)``. The function writes
``silu(gate) * up`` into ``out`` and returns ``None``. Positive ``limit`` enables
the clipping behavior described by the kernel and its reference test.

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # BEGIN activation
   :end-before: # END activation

.. aiter-function:: aiter.ops.activation.gelu_and_mul

.. aiter-function:: aiter.ops.activation.gelu_tanh_and_mul

.. aiter-function:: aiter.ops.activation.gelu_fast

The GELU-and-multiply variants split their input like the gated SiLU operator.
``gelu_fast`` instead writes a same-shape activation. See
`test_activation.py <https://github.com/ROCm/aiter/blob/main/op_tests/test_activation.py>`_
for dtype, reference and benchmark cases.

Other operator families
-----------------------

* `RoPE <https://github.com/ROCm/aiter/blob/main/aiter/ops/rope.py>`_ has separate
  cached, position-indexed and packed-layout interfaces. Use the matching
  ``op_tests/test_rope.py`` cases rather than a generic two-tensor wrapper.
* `Quantization <https://github.com/ROCm/aiter/blob/main/aiter/ops/quant.py>`_
  defines tensor, token and block scale contracts; scales and packed layouts
  are part of the API.
* `Sampling <https://github.com/ROCm/aiter/tree/main/aiter/ops>`_ is provided by
  specific sampling modules. Their probability/logit inputs and workspace
  requirements must match the operator test.
* :doc:`../triton_comms` describes the optional Iris communication path.

Hardware and dtype support is operator-specific. Benchmark only after validating
numerical output with the actual layout and backend used by your application.

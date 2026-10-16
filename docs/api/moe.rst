Mixture of Experts
==================

Use ``from aiter.fused_moe import fused_topk, fused_moe`` for the high-level
local-expert path. :doc:`../quickstart` includes a complete BF16 example with a
reference comparison. Signatures below are generated from the current source.

Routing
-------

.. aiter-function:: aiter.fused_moe.fused_topk

``hidden_states`` and ``gating_output`` have the same token count. The result is
``(topk_weights, topk_ids)`` with shape ``(tokens, topk)``; IDs are int32 and
routing weights float32. ``renormalize=True`` normalizes the selected weights.

Expert computation
------------------

.. aiter-function:: aiter.fused_moe.fused_moe

For unquantized SiLU gate/up experts, ``w1`` has logical shape
``(experts, 2 * intermediate, hidden)`` and ``w2`` has shape
``(experts, hidden, intermediate)``. Routing is supplied explicitly; this
function does not accept router logits as its fourth argument. The result is
``(tokens, hidden)``. Weight shuffling, quantization scales, expert masks and
optional output buffers must follow the selected backend's contract.

`test_moe_2stage.py <https://github.com/ROCm/aiter/blob/main/op_tests/test_moe_2stage.py>`_
contains the maintained quantization, routing and output-buffer cases. Use these
for supported combinations rather than extending the simple example by changing
only a dtype. Expert-parallel transport and synchronization belong to the
framework or communication layer.

Low-level assembly entry point
------------------------------

.. aiter-function:: aiter.ops.moe_op.fmoe

``fmoe`` is an output-buffer API that consumes sorted routing metadata and
returns ``None``. It is not an allocating wrapper around router logits. Prefer
the higher-level interface unless you are integrating the matching sorting and
assembly kernel path yourself.

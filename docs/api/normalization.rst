Normalization
=============

Allocating RMSNorm
------------------

.. aiter-function:: aiter.ops.rmsnorm.rms_norm

Use ``from aiter.ops.rmsnorm import rms_norm``. Supply an input matrix
``(tokens, hidden)`` and a weight vector ``(hidden,)`` on the same GPU.
``epsilon`` is required. The allocating interface dispatches by dtype, shape
and target; its common HIP path handles FP16/BF16 inputs. Other paths have
additional architecture restrictions, so source availability does not promise
FP32 or T5 support on every target. See :doc:`../quickstart` for a BF16
reference comparison and `test_rmsnorm2d.py
<https://github.com/ROCm/aiter/blob/main/op_tests/test_rmsnorm2d.py>`_ for more cases.

Output-buffer RMSNorm
---------------------

.. aiter-function:: aiter.ops.rmsnorm.rmsnorm

This distinct low-level function writes into ``out`` and returns ``None``.
Pass ``epsilon``, not ``eps``. Allocate output with the intended device, shape
and dtype before calling it.

Layer normalization
-------------------

.. aiter-function:: aiter.ops.norm.layer_norm

.. aiter-function:: aiter.ops.norm.layernorm2d_fwd

These are ``layer_norm`` / ``layernorm2d_fwd``, not ``aiter.layernorm``. Follow
`test_layernorm2d.py <https://github.com/ROCm/aiter/blob/main/op_tests/test_layernorm2d.py>`_
for input, weight, bias and epsilon construction. Fused residual and quantized
variants have separate mutation and scale contracts in the source module.

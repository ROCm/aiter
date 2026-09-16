Quickstart
==========

Install a matched ROCm/PyTorch/AITER stack using :doc:`installation`. These
examples target CDNA gfx942/gfx950 with FP16 attention and BF16 normalization
and MoE. They are small reference checks, not a benchmark or a statement that
every release/hardware combination has been validated. First use may JIT
compile kernels. On other targets, consult the operator's tests and restrictions.

From a source checkout, run the complete example:

.. code-block:: bash

   python docs/examples/quickstart.py

The script logs the package, PyTorch, HIP and GPU versions, raises on a
numerical mismatch, and prints each output shape on success. Record its output
and ``git rev-parse HEAD`` when validating a stack. The CPU documentation job
checks Python syntax and source signatures; execute this script on your
supported ROCm stack to validate imports, dispatch and numerical results.

Flash attention
---------------

Inputs use **BSHD** layout: batch, sequence, heads, head dimension. The FP32
reference implements the same causal mask and scale.

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-after: # BEGIN attention
   :end-before: # END attention

Packed variable-length attention
--------------------------------

Packed tokens use ``(total_tokens, heads, head_dim)`` with device-side int32
cumulative sequence offsets. This example checks each sequence independently;
it does not configure a paged KV cache.

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-after: # BEGIN varlen
   :end-before: # END varlen

RMSNorm
-------

``rms_norm`` allocates its result. The similarly named low-level ``rmsnorm``
requires an output buffer; see :doc:`api/normalization`.

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-after: # BEGIN rmsnorm
   :end-before: # END rmsnorm

Mixture of experts
------------------

Route tokens first, then call ``aiter.fused_moe.fused_moe`` with routing weights
and IDs. Gate/up weights are ``(experts, 2 * intermediate, hidden)``; down weights
are ``(experts, hidden, intermediate)``. This example follows the unquantized
path in ``op_tests/test_moe_2stage.py`` and compares with its maintained PyTorch
reference. Quantized paths require their own packing and scale contracts.

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-after: # BEGIN moe
   :end-before: # END moe

Next steps
----------

* :doc:`api/attention`, :doc:`api/gemm`, :doc:`api/moe`, :doc:`api/normalization`
* :doc:`tutorials/basic_usage`: timing and memory measurement
* :doc:`autotuning_pipeline`: matched configuration tuning and validation

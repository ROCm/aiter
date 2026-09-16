GEMM Operations
===============

AITER exposes precision- and layout-specific GEMM entry points. The signatures
below are read from source, so removed functions or renamed parameters fail the
documentation build. These are not generic ``torch.matmul`` replacements: use
the corresponding operator test to construct scales, strides and packed weights.

FP16/BF16 assembly GEMM
-----------------------

.. aiter-function:: aiter.ops.gemm_op_a16w16.gemm_a16w16_asm

The caller supplies ``out``. Logical operands are ``A: (M, K)`` and
``B: (N, K)`` for ``A @ B.T``; ``bpreshuffle=True`` requires the backend's
preshuffled weight layout. Kernel and target constraints are covered in
`test_gemm_a16w16.py <https://github.com/ROCm/aiter/blob/main/op_tests/test_gemm_a16w16.py>`_.
Do not share an ASM SplitK semaphore between concurrently executing streams;
the wrapper manages its workspace per device and stream.

Quantized GEMM
--------------

.. aiter-function:: aiter.ops.gemm_op_a8w8.gemm_a8w8

``XQ: (M, K)`` and ``WQ: (N, K)`` are quantized operands; ``x_scale`` and
``w_scale`` must match the selected quantization scheme. This allocating wrapper
chooses a backend by target. See `test_gemm_a8w8.py
<https://github.com/ROCm/aiter/blob/main/op_tests/test_gemm_a8w8.py>`_ for scale
construction and the supported input/output types.

.. aiter-function:: aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale

.. aiter-function:: aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle

Block scales and preshuffled weights are separate contracts, not flags that can
be applied to arbitrary tensors. Start from the corresponding tests and tuning
inputs; see :doc:`../autotuning_pipeline` before adding a tuned configuration.

Batched BF16 GEMM
-----------------

.. aiter-function:: aiter.ops.batched_gemm_op_bf16.batched_gemm_bf16

The caller provides ``out``; layout and stride restrictions belong to this
specific CK operator. See `test_batched_gemm_bf16.py
<https://github.com/ROCm/aiter/blob/main/op_tests/test_batched_gemm_bf16.py>`_ and its source binding.

MoE uses the routing-aware interface in :doc:`moe`. The previously listed
generic ``grouped_gemm``, ``batched_gemm``, ``gemm_bias``, ``gemm_gelu``,
``gemm_relu``, ``cutlass_gemm``, ``sparse_gemm`` and ``int8_gemm`` names are not
AITER public functions. Do not infer support from those names or from an
unqualified speedup table.

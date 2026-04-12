GEMM Operations
===============

AITER provides optimized General Matrix Multiply (GEMM) operations for AMD GPUs
across multiple precisions (FP8, BF16/FP16, FP4) with multiple backend
implementations (ASM, CK, CK Tile, Triton, FlyDSL).


A8W8 (FP8) GEMM
----------------

Functions in ``aiter.ops.gemm_op_a8w8``. All functions compute
``Out = dequant(XQ @ WQ^T)`` using FP8 (8-bit floating point) inputs with
per-tensor or block-wise scaling.

.. function:: gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out, bias=None, splitK=0)

   CK (Composable Kernel) based FP8 GEMM with per-tensor scaling.

   :param XQ: Activation tensor ``[M, K]``, FP8.
   :param WQ: Weight tensor ``[N, K]``, FP8.
   :param x_scale: Activation scale ``[M, 1]``, FP32.
   :param w_scale: Weight scale ``[1, N]``, FP32.
   :param Out: Pre-allocated output tensor ``[M, N]``.
   :param bias: Optional bias tensor.
   :param splitK: Split-K factor for parallelism (default 0 = auto).
   :returns: Output tensor ``Out``.

.. function:: gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Out, splitK=0)

   CK-based FP8 GEMM with pre-shuffled weight layout for improved memory access.

   :param XQ: Activation tensor ``[M, K]``, FP8.
   :param WQ: Pre-shuffled weight tensor ``[N, K]``, FP8.
   :param x_scale: Activation scale.
   :param w_scale: Weight scale.
   :param Out: Pre-allocated output tensor ``[M, N]``.
   :param splitK: Split-K factor (default 0).
   :returns: Output tensor ``Out``.

.. function:: gemm_a8w8_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, out, splitK=0)

   CK Tile variant of FP8 GEMM with pre-shuffled weights.

   Same interface as ``gemm_a8w8_bpreshuffle_ck``.

.. function:: gemm_a8w8_bpreshuffle_flydsl(XQ, WQ, x_scale, w_scale, Out, config)

   FlyDSL variant of FP8 GEMM with pre-shuffled weights. Falls back to
   ``gemm_a8w8_bpreshuffle_ck`` if no matching FlyDSL kernel is found.

   :param config: Dictionary with ``kernelId`` selecting the FlyDSL kernel.

.. function:: gemm_a8w8_asm(XQ, WQ, x_scale, w_scale, Out, kernelName="", bias=None, bpreshuffle=True, splitK=None)

   ASM (hand-tuned assembly) FP8 GEMM. Highest performance for supported shapes.

   :param XQ: Activation tensor ``[M, K]``, INT8/FP8.
   :param WQ: Weight tensor ``[N, K]``, shuffled layout ``(32, 16)``.
   :param x_scale: Activation scale ``[M, 1]``, FP32.
   :param w_scale: Weight scale ``[1, N]``, FP32.
   :param Out: Pre-allocated output tensor ``[M, N]``, BF16.
   :param kernelName: Specific ASM kernel name (empty string = auto).
   :param bias: Optional bias tensor ``[1, N]``, FP32.
   :param bpreshuffle: Whether weights are pre-shuffled (default True).
   :param splitK: Split-K factor (default None = auto).
   :returns: Output tensor ``Out``.

Block-Scale FP8 GEMM
^^^^^^^^^^^^^^^^^^^^^

Block-scale variants use per-block quantization scales instead of per-tensor
scales, providing better accuracy for large matrices.

.. function:: gemm_a8w8_blockscale_ck(XQ, WQ, x_scale, w_scale, Out)

   CK-based block-scale FP8 GEMM.

.. function:: gemm_a8w8_blockscale_cktile(XQ, WQ, x_scale, w_scale, Out, isBpreshuffled=False)

   CK Tile variant of block-scale FP8 GEMM.

   :param isBpreshuffled: Whether the weight tensor uses pre-shuffled layout.

.. function:: gemm_a8w8_blockscale_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Out)

   CK-based block-scale FP8 GEMM with pre-shuffled weights.

.. function:: gemm_a8w8_blockscale_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Out, isBpreshuffled=True)

   CK Tile variant of block-scale FP8 GEMM with pre-shuffled weights.

.. function:: gemm_a8w8_blockscale_bpreshuffle_asm(A, B, out, A_scale, B_scale, bias=None, splitK=None, kernelName=None, bpreshuffle=True, zero_bias_buf=None)

   ASM block-scale FP8 GEMM with pre-shuffled weights.

   :param zero_bias_buf: Optional zero-initialized bias buffer ``[1, N]``, FP32.
     Auto-created if both ``bias`` and ``zero_bias_buf`` are None.

.. function:: flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, out)

   ASM block-scale FP8 flat matrix multiply.


A16W16 (BF16/FP16) GEMM
------------------------

Functions in ``aiter.ops.gemm_op_a16w16``.

.. function:: gemm_a16w16_asm(A, B, out, bias=None, splitK=None, kernelName=None, bpreshuffle=False)

   ASM-optimized BF16/FP16 GEMM.

   :param A: Activation tensor ``[M, K]``, BF16/FP16.
   :param B: Weight tensor ``[N, K]``, BF16/FP16.
   :param out: Pre-allocated output tensor ``[M, N]``.
   :param bias: Optional bias tensor.
   :param splitK: Split-K factor (default None = auto).
   :param kernelName: Specific ASM kernel name.
   :param bpreshuffle: Whether weights are pre-shuffled (default False).
   :returns: Output tensor ``out``.


A4W4 (FP4) GEMM
----------------

Functions in ``aiter.ops.gemm_op_a4w4``. These operate on MXFP4 (4-bit
floating point) packed inputs where each byte holds two FP4 values.

.. note::

   A4W4 GEMM is **not supported** on gfx942 (MI300X). Supported on gfx950+.

.. function:: gemm_a4w4(A, B, A_scale, B_scale, bias=None, dtype=torch.bfloat16, alpha=1.0, beta=0.0, bpreshuffle=True)

   Top-level FP4 GEMM. Auto-selects between ``gemm_a4w4_blockscale`` and
   ``gemm_a4w4_asm`` based on tuned configuration.

   :param A: Activation tensor ``[M, K/2]``, packed FP4x2.
   :param B: Weight tensor ``[N, K/2]``, packed FP4x2.
   :param A_scale: Activation scale ``[M, K/32]``, E8M0 format.
   :param B_scale: Weight scale ``[N, K/32]``, E8M0 format.
   :param bias: Optional bias tensor ``[1, N]``, FP32.
   :param dtype: Output dtype (default BF16).
   :param alpha: Scalar multiplier (default 1.0).
   :param beta: Accumulation scalar (default 0.0).
   :param bpreshuffle: Whether weights are pre-shuffled (default True).
   :returns: Output tensor ``[M, N]`` in ``dtype``.

.. function:: gemm_a4w4_asm(A, B, A_scale, B_scale, out, kernelName="", bias=None, alpha=1.0, beta=0.0, bpreshuffle=True, log2_k_split=None)

   ASM-optimized FP4 GEMM.

   :param out: Pre-allocated output tensor. Dim0 must be padded to multiples of 32.
   :param log2_k_split: Log2 of the K-split factor.

.. function:: gemm_a4w4_blockscale(XQ, WQ, x_scale, w_scale, Out, splitK=0)

   CK-based block-scale FP4 GEMM.


Batched GEMM
-------------

Batched GEMM operations for processing multiple independent matrix multiplications
in a single kernel launch.

Batched FP8 GEMM
^^^^^^^^^^^^^^^^^

Functions in ``aiter.ops.batched_gemm_op_a8w8``.

.. function:: batched_gemm_a8w8(XQ, WQ, x_scale, w_scale, out, bias=None, splitK=0)

   Low-level batched FP8 GEMM. Requires pre-allocated output tensor.

   :param XQ: Batched activation tensor ``[B, M, K]``, FP8.
   :param WQ: Batched weight tensor ``[B, K, N]``, FP8.
   :param x_scale: Activation scale tensor.
   :param w_scale: Weight scale tensor.
   :param out: Pre-allocated output tensor ``[B, M, N]``.
   :param bias: Optional bias tensor.
   :param splitK: Split-K factor (default 0).
   :returns: Output tensor ``out``.

.. function:: batched_gemm_a8w8_CK(XQ, WQ, x_scale, w_scale, bias=None, dtype=torch.bfloat16, splitK=None)

   High-level batched FP8 GEMM with CK tuning. Auto-allocates output tensor and
   selects optimal split-K from tuned configuration.

   :param dtype: Output dtype (BF16 or FP16).
   :returns: Output tensor ``[B, M, N]``.

Batched BF16 GEMM
^^^^^^^^^^^^^^^^^^

Functions in ``aiter.ops.batched_gemm_op_bf16``.

.. function:: batched_gemm_bf16(XQ, WQ, out, bias=None, splitK=0)

   Low-level batched BF16 GEMM.

   :param XQ: Batched activation tensor ``[B, M, K]``, BF16.
   :param WQ: Batched weight tensor ``[B, K, N]``, BF16.
   :param out: Pre-allocated output tensor ``[B, M, N]``.
   :param bias: Optional bias tensor.
   :param splitK: Split-K factor (default 0).
   :returns: Output tensor ``out``.

.. function:: batched_gemm_bf16_CK(XQ, WQ, bias=None, dtype=torch.bfloat16, splitK=None)

   High-level batched BF16 GEMM with CK tuning. Auto-allocates output.

   :param dtype: Output dtype (BF16 or FP16).
   :returns: Output tensor ``[B, M, N]``.


DeepGEMM
--------

Functions in ``aiter.ops.deepgemm``. DeepGEMM provides grouped GEMM operations
with explicit group layout control.

.. function:: deepgemm(XQ, WQ, Y, group_layout, x_scale=None, w_scale=None)

   Top-level DeepGEMM entry point. Currently delegates to ``deepgemm_ck``.

   :param XQ: Activation tensor.
   :param WQ: Weight tensor.
   :param Y: Pre-allocated output tensor.
   :param group_layout: Tensor describing the group layout.
   :param x_scale: Optional activation scale.
   :param w_scale: Optional weight scale.
   :returns: Output tensor ``Y``.

.. function:: deepgemm_ck(XQ, WQ, Y, group_layout, x_scale=None, w_scale=None)

   CK-based DeepGEMM implementation.


Auto-Tuned GEMM
----------------

Functions in ``aiter.tuned_gemm``. The auto-tuning layer selects the best
backend (ASM, CK, hipBLASLt, Triton, FlyDSL, or PyTorch) based on matrix shape
and pre-computed tuning configurations.

.. function:: gemm_a16w16(A, B, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None)

   Top-level BF16/FP16 GEMM with automatic backend selection.

   The backend is chosen from tuned CSV configurations keyed on ``(M, N, K, dtype)``.
   Supported backends: ``hipblaslt``, ``asm``, ``skinny``, ``triton``, ``flydsl``,
   ``torch`` (fallback).

   :param A: Activation tensor ``[M, K]`` (or ``[*, M, K]`` for batched).
   :param B: Weight tensor ``[N, K]``.
   :param bias: Optional bias tensor.
   :param otype: Output dtype (default same as input).
   :param scale_a: Optional activation scale (for FP8 inputs via hipBLASLt).
   :param scale_b: Optional weight scale.
   :param scale_c: Optional output scale.
   :returns: Output tensor ``[M, N]``.

.. class:: TunedGemm

   Stateful wrapper around ``gemm_a16w16`` for BF16/FP16 GEMM with optional
   FP8 per-tensor quantization scaling.

   .. method:: mm(inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None)

      Delegates to :func:`gemm_a16w16`.

   A global instance ``tgemm`` is available as ``aiter.tuned_gemm.tgemm``.


Backend Selection
-----------------

AITER provides multiple backend implementations for each precision:

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Backend
     - Description
     - Used By
   * - **CK**
     - AMD Composable Kernel library. Default for most shapes.
     - A8W8, A4W4, Batched, DeepGEMM
   * - **CK Tile**
     - Tile-based CK variant with different tiling strategies.
     - A8W8 block-scale and pre-shuffle
   * - **ASM**
     - Hand-tuned GFX ISA assembly. Best peak performance.
     - A8W8, A16W16, A4W4, block-scale
   * - **FlyDSL**
     - AMD FlyDSL code-generation framework.
     - A8W8 pre-shuffle, A16W16
   * - **Triton**
     - OpenAI Triton for AMD GPUs. Portable, CK-free path.
     - A16W16
   * - **hipBLASLt**
     - AMD hipBLASLt library (via ``hipb_mm``).
     - A16W16 (gfx942)
   * - **PyTorch**
     - ``torch.nn.functional.linear`` fallback.
     - A16W16 (untuned shapes)

For production inference, use :func:`gemm_a16w16` or :class:`TunedGemm` which
automatically select the fastest backend. Use precision-specific functions
(``gemm_a8w8_*``, ``gemm_a4w4_*``) when you need explicit control over the
quantization format and backend.


See Also
--------

* :doc:`moe` - MoE-specific grouped GEMM operations

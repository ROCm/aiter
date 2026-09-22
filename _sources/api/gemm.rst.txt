GEMM Operations
===============

AITER provides optimized General Matrix Multiply (GEMM) operations for AMD GPUs.

Grouped GEMM
------------

.. autofunction:: aiter.grouped_gemm

Efficient grouped matrix multiplication for Mixture of Experts (MoE) layers.

**Parameters:**

* **input** (*torch.Tensor*) - Input tensor ``(total_tokens, hidden_dim)``
* **weights** (*torch.Tensor*) - Expert weights ``(num_experts, hidden_dim, output_dim)``
* **expert_ids** (*torch.Tensor*) - Expert assignments ``(total_tokens, top_k)``
* **topk** (*int*, optional) - Number of experts per token. Default: inferred from ``expert_ids``

**Returns:**

* **output** (*torch.Tensor*) - Result ``(total_tokens, output_dim)``

**Example:**

.. code-block:: python

   import torch
   import aiter

   # 4096 tokens, 512 hidden dim, 8 experts, top-2 routing
   tokens = 4096
   hidden = 512
   num_experts = 8
   output_dim = 2048

   x = torch.randn(tokens, hidden, device='cuda', dtype=torch.float16)
   expert_weights = torch.randn(num_experts, hidden, output_dim,
                                device='cuda', dtype=torch.float16)
   expert_ids = torch.randint(0, num_experts, (tokens, 2), device='cuda')

   output = aiter.grouped_gemm(x, expert_weights, expert_ids)

Batched GEMM
------------

.. autofunction:: aiter.batched_gemm

Batched matrix multiplication with optimizations for AMD hardware.

**Parameters:**

* **a** (*torch.Tensor*) - First batch ``(batch, m, k)``
* **b** (*torch.Tensor*) - Second batch ``(batch, k, n)``
* **transpose_a** (*bool*, optional) - Transpose A. Default: ``False``
* **transpose_b** (*bool*, optional) - Transpose B. Default: ``False``

**Returns:**

* **output** (*torch.Tensor*) - Result ``(batch, m, n)``

Fused GEMM Operations
---------------------

GEMM + Bias
^^^^^^^^^^^

.. autofunction:: aiter.gemm_bias

Matrix multiply with bias addition fused.

**Parameters:**

* **input** (*torch.Tensor*) - Input ``(m, k)``
* **weight** (*torch.Tensor*) - Weight ``(k, n)`` or ``(n, k)`` if transposed
* **bias** (*torch.Tensor*) - Bias ``(n,)``
* **transpose_weight** (*bool*, optional) - Default: ``True``

**Returns:**

* **output** (*torch.Tensor*) - ``(m, n)``

**Example:**

.. code-block:: python

   x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
   weight = torch.randn(2048, 512, device='cuda', dtype=torch.float16)
   bias = torch.randn(2048, device='cuda', dtype=torch.float16)

   # Fused: y = x @ weight.T + bias
   output = aiter.gemm_bias(x, weight, bias, transpose_weight=True)

GEMM + GELU
^^^^^^^^^^^

.. autofunction:: aiter.gemm_gelu

Matrix multiply with GELU activation fused.

**Parameters:**

* **input** (*torch.Tensor*) - Input tensor
* **weight** (*torch.Tensor*) - Weight tensor
* **bias** (*torch.Tensor*, optional) - Bias tensor

**Returns:**

* **output** (*torch.Tensor*) - Result with GELU applied

GEMM + ReLU
^^^^^^^^^^^

.. autofunction:: aiter.gemm_relu

Matrix multiply with ReLU activation fused.

**Parameters:**

* **input** (*torch.Tensor*) - Input tensor
* **weight** (*torch.Tensor*) - Weight tensor
* **bias** (*torch.Tensor*, optional) - Bias tensor

**Returns:**

* **output** (*torch.Tensor*) - Result with ReLU applied

CUTLASS-style GEMM
------------------

.. autofunction:: aiter.cutlass_gemm

High-performance GEMM using CUTLASS-inspired kernels for AMD.

**Parameters:**

* **a** (*torch.Tensor*) - Matrix A
* **b** (*torch.Tensor*) - Matrix B
* **alpha** (*float*, optional) - Scalar multiplier. Default: ``1.0``
* **beta** (*float*, optional) - Scalar for accumulation. Default: ``0.0``
* **c** (*torch.Tensor*, optional) - Accumulation matrix

**Returns:**

* **output** (*torch.Tensor*) - Result: ``alpha * (A @ B) + beta * C``

Sparse GEMM
-----------

.. autofunction:: aiter.sparse_gemm

Sparse matrix multiplication with various sparsity patterns.

**Parameters:**

* **input** (*torch.Tensor*) - Dense input
* **weight** (*torch.Tensor*) - Sparse weight (CSR/COO format)
* **sparsity_pattern** (*str*) - Pattern type: ``'csr'``, ``'coo'``, ``'block'``

**Returns:**

* **output** (*torch.Tensor*) - Dense output

INT8 Quantized GEMM
-------------------

.. autofunction:: aiter.int8_gemm

INT8 quantized matrix multiplication for inference acceleration.

**Parameters:**

* **input** (*torch.Tensor*) - Quantized input INT8
* **weight** (*torch.Tensor*) - Quantized weight INT8
* **input_scale** (*torch.Tensor*) - Input dequantization scale
* **weight_scale** (*torch.Tensor*) - Weight dequantization scale
* **output_dtype** (*torch.dtype*, optional) - Output type. Default: ``torch.float16``

**Returns:**

* **output** (*torch.Tensor*) - Dequantized result

**Example:**

.. code-block:: python

   # Quantized matrices (simulated)
   x_int8 = torch.randint(-127, 127, (1024, 512), device='cuda', dtype=torch.int8)
   w_int8 = torch.randint(-127, 127, (2048, 512), device='cuda', dtype=torch.int8)

   # Scales for dequantization
   x_scale = torch.randn(1, device='cuda', dtype=torch.float32)
   w_scale = torch.randn(1, device='cuda', dtype=torch.float32)

   # INT8 GEMM with automatic dequantization
   output = aiter.int8_gemm(x_int8, w_int8, x_scale, w_scale)

Performance Characteristics
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Operation
     - Typical Speedup
     - Best Use Case
     - Memory Usage
   * - grouped_gemm
     - 3-10x vs loops
     - MoE layers
     - Low
   * - batched_gemm
     - 2-4x vs sequential
     - Batched inference
     - Moderate
   * - gemm_bias
     - 1.5-2x vs unfused
     - Linear layers
     - Low
   * - int8_gemm
     - 2-3x vs FP16
     - Quantized models
     - Very Low
   * - cutlass_gemm
     - Best raw GEMM
     - Large matrices
     - Moderate

Optimization Tips
-----------------

1. **Matrix Dimensions**: Multiple of 128 for best performance (MI300X)
2. **Data Layout**: Row-major preferred for AMD GPUs
3. **Precision**: FP16/BF16 recommended over FP32
4. **Fusion**: Use fused ops (gemm_bias, gemm_gelu) when possible
5. **Batch Size**: Larger batches improve throughput

Example: Optimal MoE Forward Pass
----------------------------------

.. code-block:: python

   import torch
   import aiter

   class OptimizedMoELayer:
       def __init__(self, hidden_dim, num_experts, expert_dim):
           self.hidden_dim = hidden_dim
           self.num_experts = num_experts
           self.expert_dim = expert_dim

           # Expert weights (all experts in one tensor)
           self.w1 = torch.randn(num_experts, hidden_dim, expert_dim,
                                device='cuda', dtype=torch.float16)
           self.w2 = torch.randn(num_experts, expert_dim, hidden_dim,
                                device='cuda', dtype=torch.float16)

       def forward(self, x, expert_ids, routing_weights):
           # x: (total_tokens, hidden_dim)
           # expert_ids: (total_tokens, top_k)
           # routing_weights: (total_tokens, top_k)

           # First grouped GEMM
           hidden = aiter.grouped_gemm(x, self.w1, expert_ids)

           # Activation
           hidden = aiter.gelu(hidden)

           # Second grouped GEMM
           output = aiter.grouped_gemm(hidden, self.w2, expert_ids)

           # Apply routing weights
           output = output * routing_weights.unsqueeze(-1)

           return output

See Also
--------

* :doc:`../tutorials/moe` - MoE tutorial
* :doc:`../tutorials/quantization` - INT8 quantization guide
* :doc:`moe` - MoE-specific operations
* :doc:`../benchmarks` - GEMM benchmarks

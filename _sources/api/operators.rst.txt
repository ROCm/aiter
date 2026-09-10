Core Operators
==============

RMSNorm
-------

.. autofunction:: aiter.rmsnorm

Root Mean Square Layer Normalization, commonly used in LLMs like Llama.

**Parameters:**

* **x** (*torch.Tensor*) - Input tensor of shape ``(..., hidden_dim)``
* **weight** (*torch.Tensor*) - Scaling weights of shape ``(hidden_dim,)``
* **eps** (*float*, optional) - Epsilon for numerical stability. Default: ``1e-6``

**Returns:**

* **output** (*torch.Tensor*) - Normalized tensor with same shape as input

**Example:**

.. code-block:: python

   import torch
   import aiter

   x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
   weight = torch.ones(4096, device='cuda', dtype=torch.float16)

   output = aiter.rmsnorm(x, weight, eps=1e-6)

LayerNorm
---------

.. autofunction:: aiter.layernorm

Standard layer normalization with optional bias.

**Parameters:**

* **x** (*torch.Tensor*) - Input tensor ``(..., hidden_dim)``
* **weight** (*torch.Tensor*) - Weights ``(hidden_dim,)``
* **bias** (*torch.Tensor*, optional) - Bias ``(hidden_dim,)``
* **eps** (*float*, optional) - Epsilon. Default: ``1e-5``

**Returns:**

* **output** (*torch.Tensor*) - Normalized output

SoftMax
-------

.. autofunction:: aiter.softmax

Optimized softmax operation with optional masking.

**Parameters:**

* **x** (*torch.Tensor*) - Input tensor
* **dim** (*int*) - Dimension to apply softmax
* **mask** (*torch.Tensor*, optional) - Attention mask

**Returns:**

* **output** (*torch.Tensor*) - Softmax output

GELU
----

.. autofunction:: aiter.gelu

Fast GELU activation function.

**Parameters:**

* **x** (*torch.Tensor*) - Input tensor
* **approximate** (*str*, optional) - Approximation method. Options: ``'none'``, ``'tanh'``. Default: ``'none'``

**Returns:**

* **output** (*torch.Tensor*) - GELU output

**Example:**

.. code-block:: python

   import torch
   import aiter

   x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)

   # Exact GELU
   output_exact = aiter.gelu(x)

   # Fast approximate GELU
   output_approx = aiter.gelu(x, approximate='tanh')

SwiGLU
------

.. autofunction:: aiter.swiglu

Swish-Gated Linear Unit activation.

**Parameters:**

* **x** (*torch.Tensor*) - Input tensor ``(..., 2 * hidden_dim)``
* **dim** (*int*, optional) - Dimension to split. Default: ``-1``

**Returns:**

* **output** (*torch.Tensor*) - SwiGLU output ``(..., hidden_dim)``

Rotary Position Embedding (RoPE)
---------------------------------

.. autofunction:: aiter.apply_rotary_pos_emb

Apply rotary position embeddings to query and key tensors.

**Parameters:**

* **q** (*torch.Tensor*) - Query tensor ``(batch, seq_len, num_heads, head_dim)``
* **k** (*torch.Tensor*) - Key tensor ``(batch, seq_len, num_heads, head_dim)``
* **cos** (*torch.Tensor*) - Cosine embeddings ``(seq_len, head_dim // 2)``
* **sin** (*torch.Tensor*) - Sine embeddings ``(seq_len, head_dim // 2)``
* **position_ids** (*torch.Tensor*, optional) - Position indices

**Returns:**

* **q_rot** (*torch.Tensor*) - Rotated query
* **k_rot** (*torch.Tensor*) - Rotated key

**Example:**

.. code-block:: python

   import torch
   import aiter

   seq_len, head_dim = 1024, 64
   q = torch.randn(2, seq_len, 16, head_dim, device='cuda', dtype=torch.float16)
   k = torch.randn(2, seq_len, 16, head_dim, device='cuda', dtype=torch.float16)

   # Precompute RoPE embeddings
   cos, sin = aiter.precompute_rope_embeddings(seq_len, head_dim)

   # Apply rotation
   q_rot, k_rot = aiter.apply_rotary_pos_emb(q, k, cos, sin)

Sampling Operations
-------------------

Top-K Sampling
^^^^^^^^^^^^^^

.. autofunction:: aiter.top_k_sampling

Sample from top-k logits.

**Parameters:**

* **logits** (*torch.Tensor*) - Logits ``(batch, vocab_size)``
* **k** (*int*) - Number of top candidates
* **temperature** (*float*, optional) - Sampling temperature. Default: ``1.0``

**Returns:**

* **tokens** (*torch.Tensor*) - Sampled token IDs ``(batch,)``

Top-P (Nucleus) Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: aiter.top_p_sampling

Nucleus sampling with probability threshold.

**Parameters:**

* **logits** (*torch.Tensor*) - Logits ``(batch, vocab_size)``
* **p** (*float*) - Cumulative probability threshold (0.0 to 1.0)
* **temperature** (*float*, optional) - Temperature. Default: ``1.0``

**Returns:**

* **tokens** (*torch.Tensor*) - Sampled tokens ``(batch,)``

Performance Notes
-----------------

All operators are optimized for AMD GPUs:

* **FP16/BF16 preferred**: Best performance on MI300X
* **Large batches**: Better GPU utilization
* **Fused operations**: Many ops fused into single kernels
* **In-place when possible**: Reduces memory allocations

Supported Data Types
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operator
     - FP32
     - FP16
     - BF16
   * - rmsnorm
     - ✓
     - ✓ (fastest)
     - ✓
   * - layernorm
     - ✓
     - ✓ (fastest)
     - ✓
   * - gelu
     - ✓
     - ✓ (fastest)
     - ✓
   * - swiglu
     - ✓
     - ✓ (fastest)
     - ✓
   * - apply_rotary_pos_emb
     - ✓
     - ✓ (fastest)
     - ✓
   * - sampling ops
     - ✓
     - ✓
     - ✓

See Also
--------

* :doc:`../tutorials/normalization` - Normalization tutorial
* :doc:`../tutorials/custom_ops` - Adding custom operators
* :doc:`gemm` - Matrix multiplication operations

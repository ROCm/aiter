Attention Operations
====================

AITER provides highly optimized attention kernels for AMD GPUs with ROCm.

Flash Attention
---------------

.. autofunction:: aiter.flash_attn_func

Standard flash attention implementation with optional causal masking.

**Parameters:**

* **query** (*torch.Tensor*) - Query tensor of shape ``(batch, seq_len, num_heads, head_dim)``
* **key** (*torch.Tensor*) - Key tensor of shape ``(batch, seq_len, num_heads, head_dim)``
* **value** (*torch.Tensor*) - Value tensor of shape ``(batch, seq_len, num_heads, head_dim)``
* **causal** (*bool*, optional) - Whether to apply causal masking. Default: ``False``
* **softmax_scale** (*float*, optional) - Scaling factor for softmax. Default: ``1/sqrt(head_dim)``

**Returns:**

* **output** (*torch.Tensor*) - Attention output of shape ``(batch, seq_len, num_heads, head_dim)``

**Example:**

.. code-block:: python

   import torch
   import aiter

   q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
   k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
   v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

   output = aiter.flash_attn_func(q, k, v, causal=True)

Flash Attention with KV Cache
------------------------------

.. autofunction:: aiter.flash_attn_with_kvcache

Optimized attention with paged KV cache support for inference.

**Parameters:**

* **query** (*torch.Tensor*) - Query tensor ``(batch, seq_len, num_heads, head_dim)``
* **kv_cache** (*torch.Tensor*) - Paged KV cache ``(num_blocks, num_heads, block_size, head_dim)``
* **page_table** (*torch.Tensor*) - Page table mapping ``(batch, max_blocks_per_seq)``
* **block_size** (*int*) - Size of each page block (e.g., 128)
* **causal** (*bool*, optional) - Causal masking. Default: ``True``

**Returns:**

* **output** (*torch.Tensor*) - Attention output ``(batch, seq_len, num_heads, head_dim)``

**Example:**

.. code-block:: python

   query = torch.randn(4, 128, 16, 64, device='cuda', dtype=torch.float16)
   kv_cache = torch.randn(256, 16, 128, 64, device='cuda', dtype=torch.float16)
   page_table = torch.randint(0, 256, (4, 32), device='cuda', dtype=torch.int32)

   output = aiter.flash_attn_with_kvcache(
       query, kv_cache, page_table, block_size=128
   )

Grouped Query Attention (GQA)
------------------------------

.. autofunction:: aiter.grouped_query_attention

Efficient grouped query attention for models like Llama 2.

**Parameters:**

* **query** (*torch.Tensor*) - ``(batch, seq_len, num_q_heads, head_dim)``
* **key** (*torch.Tensor*) - ``(batch, seq_len, num_kv_heads, head_dim)``
* **value** (*torch.Tensor*) - ``(batch, seq_len, num_kv_heads, head_dim)``
* **num_groups** (*int*) - Number of query heads per KV head
* **causal** (*bool*, optional) - Causal masking. Default: ``False``

**Returns:**

* **output** (*torch.Tensor*) - ``(batch, seq_len, num_q_heads, head_dim)``

Multi-Query Attention (MQA)
----------------------------

.. autofunction:: aiter.multi_query_attention

Multi-query attention where all query heads share single key/value heads.

**Parameters:**

* **query** (*torch.Tensor*) - ``(batch, seq_len, num_heads, head_dim)``
* **key** (*torch.Tensor*) - ``(batch, seq_len, 1, head_dim)``
* **value** (*torch.Tensor*) - ``(batch, seq_len, 1, head_dim)``
* **causal** (*bool*, optional) - Causal masking. Default: ``False``

**Returns:**

* **output** (*torch.Tensor*) - ``(batch, seq_len, num_heads, head_dim)``

Variable Sequence Attention
----------------------------

.. autofunction:: aiter.variable_length_attention

Attention with variable-length sequences using page tables.

**Parameters:**

* **query** (*torch.Tensor*) - Query tensor
* **key** (*torch.Tensor*) - Key tensor
* **value** (*torch.Tensor*) - Value tensor
* **seq_lengths** (*torch.Tensor*) - Actual sequence lengths ``(batch,)``
* **max_seq_len** (*int*) - Maximum sequence length

**Returns:**

* **output** (*torch.Tensor*) - Attention output

Supported Architectures
------------------------

AITER attention kernels are optimized for:

* **AMD Instinct MI300X** (gfx942) - Best performance
* **AMD Instinct MI250X** (gfx90a) - Fully supported
* **AMD Instinct MI300A** (gfx950) - Experimental

Performance Characteristics
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Operation
     - Typical Speedup
     - Memory Efficient
     - Best For
   * - flash_attn_func
     - 2-4x vs PyTorch
     - Yes
     - Training & Inference
   * - flash_attn_with_kvcache
     - 3-6x vs naive
     - Yes
     - LLM Inference
   * - grouped_query_attention
     - 2-3x vs unfused
     - Moderate
     - Llama-style models
   * - variable_length_attention
     - 4-8x vs padded
     - High
     - Variable batches

See Also
--------

* :doc:`../tutorials/attention` - Attention tutorial
* :doc:`../tutorials/variable_length` - Variable-length sequences
* :doc:`../benchmarks` - Performance benchmarks

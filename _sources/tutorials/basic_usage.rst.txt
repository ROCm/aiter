Basic Usage
===========

This tutorial covers the fundamental operations in AITER.

Installation Check
------------------

First, verify your installation:

.. code-block:: python

   import torch
   import aiter

   # Check versions
   print(f"PyTorch: {torch.__version__}")
   print(f"AITER: {aiter.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip}")

   # Check GPU
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
       print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

Expected output:

.. code-block:: text

   PyTorch: 2.2.0+rocm5.7
   AITER: 0.1.0
   CUDA available: True
   ROCm version: 5.7.1
   GPU: AMD Instinct MI300X
   Compute capability: (9, 4)

Hello World: Flash Attention
-----------------------------

Let's start with a simple flash attention example:

.. code-block:: python

   import torch
   import aiter

   # Set device
   device = torch.device('cuda')
   dtype = torch.float16

   # Create input tensors
   batch_size = 2
   seq_len = 512
   num_heads = 8
   head_dim = 64

   query = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device=device, dtype=dtype)
   key = torch.randn(batch_size, seq_len, num_heads, head_dim,
                     device=device, dtype=dtype)
   value = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device=device, dtype=dtype)

   # Run flash attention
   output = aiter.flash_attn_func(query, key, value, causal=True)

   print(f"Input shape: {query.shape}")
   print(f"Output shape: {output.shape}")
   print(f"Output dtype: {output.dtype}")

Understanding the Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``query, key, value``: Input tensors in BHSD layout (Batch, Seq, Heads, Dim)
* ``causal=True``: Apply causal masking (for autoregressive models)
* ``softmax_scale``: Defaults to ``1/sqrt(head_dim)`` for stability

Comparing with PyTorch
^^^^^^^^^^^^^^^^^^^^^^

Let's compare AITER with standard PyTorch attention:

.. code-block:: python

   import torch
   import torch.nn.functional as F
   import aiter
   import time

   # Setup
   batch_size, seq_len, num_heads, head_dim = 4, 1024, 16, 64
   device = torch.device('cuda')
   dtype = torch.float16

   q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device=device, dtype=dtype)
   k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device=device, dtype=dtype)
   v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device=device, dtype=dtype)

   # Warmup
   for _ in range(10):
       _ = aiter.flash_attn_func(q, k, v)
   torch.cuda.synchronize()

   # Benchmark AITER
   start = time.time()
   for _ in range(100):
       out_aiter = aiter.flash_attn_func(q, k, v)
   torch.cuda.synchronize()
   aiter_time = (time.time() - start) / 100

   # Naive PyTorch implementation
   def pytorch_attention(q, k, v):
       # Transpose to (B, H, S, D)
       q = q.transpose(1, 2)
       k = k.transpose(1, 2)
       v = v.transpose(1, 2)

       # Attention scores
       scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
       attn = F.softmax(scores, dim=-1)
       out = torch.matmul(attn, v)

       # Back to (B, S, H, D)
       return out.transpose(1, 2)

   # Benchmark PyTorch
   start = time.time()
   for _ in range(100):
       out_pytorch = pytorch_attention(q, k, v)
   torch.cuda.synchronize()
   pytorch_time = (time.time() - start) / 100

   print(f"AITER time: {aiter_time*1000:.2f} ms")
   print(f"PyTorch time: {pytorch_time*1000:.2f} ms")
   print(f"Speedup: {pytorch_time/aiter_time:.2f}x")

   # Verify correctness
   max_diff = (out_aiter - out_pytorch).abs().max()
   print(f"Max difference: {max_diff:.6f}")

Expected output:

.. code-block:: text

   AITER time: 0.45 ms
   PyTorch time: 1.82 ms
   Speedup: 4.04x
   Max difference: 0.000122

Working with Different Precisions
----------------------------------

AITER supports FP32, FP16, and BF16:

.. code-block:: python

   import torch
   import aiter

   # Test different dtypes
   dtypes = [torch.float32, torch.float16, torch.bfloat16]

   for dtype in dtypes:
       q = torch.randn(2, 512, 8, 64, device='cuda', dtype=dtype)
       k = torch.randn(2, 512, 8, 64, device='cuda', dtype=dtype)
       v = torch.randn(2, 512, 8, 64, device='cuda', dtype=dtype)

       output = aiter.flash_attn_func(q, k, v)

       print(f"{dtype}: ✓ Output shape {output.shape}")

Output:

.. code-block:: text

   torch.float32: ✓ Output shape torch.Size([2, 512, 8, 64])
   torch.float16: ✓ Output shape torch.Size([2, 512, 8, 64])
   torch.bfloat16: ✓ Output shape torch.Size([2, 512, 8, 64])

RMSNorm Example
---------------

Normalization is critical for LLMs. Here's RMSNorm:

.. code-block:: python

   import torch
   import aiter

   # Input tensor
   batch_size, seq_len, hidden_dim = 2, 1024, 4096
   x = torch.randn(batch_size, seq_len, hidden_dim,
                   device='cuda', dtype=torch.float16)

   # Normalization weight
   weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)

   # Apply RMSNorm
   output = aiter.rmsnorm(x, weight, eps=1e-6)

   print(f"Input shape: {x.shape}")
   print(f"Output shape: {output.shape}")

   # Verify normalization (should be close to 1.0)
   rms = torch.sqrt((output ** 2).mean(dim=-1))
   print(f"Output RMS (should be ~1.0): {rms.mean():.4f}")

Batched Operations
------------------

AITER is optimized for batched operations:

.. code-block:: python

   import torch
   import aiter

   # Multiple batch sizes
   batch_sizes = [1, 4, 16, 64]

   for bs in batch_sizes:
       q = torch.randn(bs, 512, 8, 64, device='cuda', dtype=torch.float16)
       k = torch.randn(bs, 512, 8, 64, device='cuda', dtype=torch.float16)
       v = torch.randn(bs, 512, 8, 64, device='cuda', dtype=torch.float16)

       # Warmup
       _ = aiter.flash_attn_func(q, k, v)
       torch.cuda.synchronize()

       # Timing
       start = torch.cuda.Event(enable_timing=True)
       end = torch.cuda.Event(enable_timing=True)

       start.record()
       output = aiter.flash_attn_func(q, k, v)
       end.record()

       torch.cuda.synchronize()
       elapsed = start.elapsed_time(end)

       print(f"Batch size {bs:3d}: {elapsed:.2f} ms "
             f"({elapsed/bs:.2f} ms/sample)")

Error Handling
--------------

Always handle errors gracefully:

.. code-block:: python

   import torch
   import aiter

   try:
       # Invalid shapes (heads dimension mismatch)
       q = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.float16)
       k = torch.randn(2, 512, 16, 64, device='cuda', dtype=torch.float16)
       v = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.float16)

       output = aiter.flash_attn_func(q, k, v)

   except RuntimeError as e:
       print(f"Error caught: {e}")

   try:
       # Wrong device (CPU not supported)
       q = torch.randn(2, 512, 8, 64, dtype=torch.float16)  # CPU tensor
       k = torch.randn(2, 512, 8, 64, dtype=torch.float16)
       v = torch.randn(2, 512, 8, 64, dtype=torch.float16)

       output = aiter.flash_attn_func(q, k, v)

   except RuntimeError as e:
       print(f"Error caught: {e}")

Memory Management
-----------------

Monitor GPU memory usage:

.. code-block:: python

   import torch
   import aiter

   # Check initial memory
   torch.cuda.reset_peak_memory_stats()
   initial_mem = torch.cuda.memory_allocated() / 1024**2

   # Create large tensors
   batch_size, seq_len, num_heads, head_dim = 8, 2048, 16, 64
   q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device='cuda', dtype=torch.float16)
   k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device='cuda', dtype=torch.float16)
   v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device='cuda', dtype=torch.float16)

   after_alloc = torch.cuda.memory_allocated() / 1024**2

   # Run attention
   output = aiter.flash_attn_func(q, k, v)
   torch.cuda.synchronize()

   peak_mem = torch.cuda.max_memory_allocated() / 1024**2

   print(f"Initial memory: {initial_mem:.1f} MB")
   print(f"After allocation: {after_alloc:.1f} MB")
   print(f"Peak memory: {peak_mem:.1f} MB")
   print(f"Memory overhead: {peak_mem - after_alloc:.1f} MB")

Next Steps
----------

* :doc:`attention_tutorial` - Deep dive into attention mechanisms
* :doc:`variable_length` - Handle variable-length sequences
* :doc:`moe_tutorial` - Mixture of Experts optimization

Common Gotchas
--------------

1. **Wrong tensor layout**: AITER expects BHSD (Batch, Seq, Heads, Dim)
2. **CPU tensors**: AITER only works with CUDA tensors
3. **Mixed precision**: Ensure all inputs have the same dtype
4. **Device mismatch**: All tensors must be on the same GPU
5. **Sequence length**: For best performance, use lengths that are multiples of 128

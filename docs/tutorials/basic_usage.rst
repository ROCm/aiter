Basic Usage
===========

Run :doc:`../quickstart` first. It includes complete attention, packed-sequence,
normalization and MoE reference checks from ``docs/examples/quickstart.py``.
For attention the layout is **BSHD** (batch, sequence, heads, dimension),
not PyTorch SDPA's BHSD layout. Use FP16/BF16 only where supported by the
specific kernel; there is no library-wide promise of FP32 attention support.

Timing GPU work
---------------

Warm up exactly the operator and shape you intend to measure. JIT compilation
and tuning can dominate the first invocation. GPU execution is asynchronous,
so CPU wall time around a launch alone is not a kernel latency measurement.
After the quickstart has constructed ``q``, ``k`` and ``v``:

.. code-block:: python

   for _ in range(10):
       flash_attn_func(q, k, v, causal=True)
   torch.cuda.synchronize()
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)
   start.record()
   for _ in range(100):
       flash_attn_func(q, k, v, causal=True)
   end.record()
   end.synchronize()
   print("milliseconds per call:", start.elapsed_time(end) / 100)

Measure a baseline with identical inputs, dtype, masking, output requirements
and warmup. Record the GPU, stack versions, AITER revision, tuning configuration
and command. Event timing includes work between the events and does not establish
end-to-end model throughput. Use the maintained ``op_tests`` benchmarks and
:doc:`../autotuning_pipeline` for larger sweeps.

Memory and correctness
----------------------

``torch.cuda.reset_peak_memory_stats()`` and
``torch.cuda.max_memory_allocated()`` report PyTorch allocator usage, not all
process/device memory. Synchronize before reading measurements, and account
for whether output allocation occurs inside or outside the timed region.

Check shapes, strides, device and dtype before invoking a kernel. A kernel may
write an output buffer and return ``None``; allocating and mutation APIs are
not interchangeable. Quantized inputs additionally need the correct scale
layout and weight packing. Reference comparisons should fail on a mismatch,
not merely print the maximum difference.

For help, attach the smallest reproduction and environment information to
`GitHub Issues <https://github.com/ROCm/aiter/issues>`_.

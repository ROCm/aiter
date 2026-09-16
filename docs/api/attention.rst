Attention Operations
====================

The signatures below are read directly from Python source without importing
AITER. Follow the linked implementation and ``op_tests/test_mha.py`` for the
supported dtype, architecture and option combinations in your checkout.

Flash attention
---------------

.. aiter-function:: aiter.ops.mha.flash_attn_func

Use ``from aiter.ops.mha import flash_attn_func`` (also exported by ``aiter``
in its normal ROCm import mode). Inputs are ``q: (B, Sq, Hq, D)``,
``k: (B, Sk, Hkv, D)`` and ``v: (B, Sk, Hkv, Dv)``. The default result is an
output tensor. ``return_lse`` and ``return_attn_probs`` change the return
structure; consult the source before unpacking optional outputs.

Set ``dropout_p=0.0`` for evaluation. The default softmax scale is
``1 / sqrt(D)``. With unequal sequence lengths, the causal mask aligns to the
**bottom right** of the attention matrix, so a square-mask example cannot be
reused unchanged for decode.

GQA and MQA use this same entry point: provide fewer KV heads, with ``Hq``
divisible by ``Hkv``. MQA is the case ``Hkv=1``. There are no separate
``grouped_query_attention`` or ``multi_query_attention`` wrappers to call.

Variable-length and paged attention
-----------------------------------

.. aiter-function:: aiter.ops.mha.flash_attn_varlen_func

The packed, non-paged form uses ``q: (total_q, Hq, D)``,
``k: (total_k, Hkv, D)`` and ``v: (total_k, Hkv, Dv)``. The cumulative offsets
are GPU int32 tensors of length ``batch + 1``; maximum sequence lengths are
Python integers. See :doc:`../quickstart` for a complete packed example.

``block_table`` selects the paged variant with a different KV layout; it is
not interchangeable with packed KV. Physical padding offsets
(``cu_seqlens_*_padded``) cannot currently be combined with ``block_table``.
Use the paged cases in `test_mha.py
<https://github.com/ROCm/aiter/blob/main/op_tests/test_mha.py>`_ as the layout
contract. AITER does not expose the previously documented
``flash_attn_with_kvcache`` wrapper.

Performance
-----------

Measure your actual shape, precision, mask and backend after warmup. Publish
the GPU, package versions, source revision and matched baseline with results;
there is no universal attention speedup independent of those choices.

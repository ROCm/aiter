# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels for Qwen sparse attention (QSA).

  ``sparse_decode_attn.py``    one CTA per (row, kv head, KV split), reading the
                               selected KV straight from the token pool
  ``sparse_decode_reduce.py``  merges the KV-split partials
  ``indexer_select.py``        scores paged index K and writes top-512 block ids
  ``sparse_gqa.py``            one CTA per (row, kv head, split) over paged K/V

The public entry point for gfx950 sparse decode is ``aiter.ops.flydsl.qsa_sparse_decode``.
Host dispatch for the indexer and sparse GQA lives in ``aiter.ops.flydsl.qsa_paged``.
The fp32 reference lives in ``op_tests/qsa_ref.py``.
"""

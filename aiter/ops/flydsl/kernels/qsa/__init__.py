# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels for Qwen sparse attention (QSA).

  ``sparse_decode_attn.py``    one CTA per (row, kv head, KV split), reading the
                               selected KV straight from the token pool
  ``sparse_decode_reduce.py``  merges the KV-split partials
  ``attention*.py``            gfx942 BF16 D256 sparse attention

The public entry points are ``aiter.ops.flydsl.qsa_sparse_decode`` (gfx950) and
``attention.qsa_sparse_attention_flydsl`` (gfx942).
"""

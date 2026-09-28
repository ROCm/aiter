# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vendored FlyDSL paged-attention decode kernels.

Sources, copied from the FlyDSL checkout with imports rewritten into
``aiter.ops.flydsl.kernels``:

  ``utils.py``          <- ``kernels/common/utils.py``     (mem_ops re-exports dropped)
  ``pa_common.py``      <- ``kernels/attention/pa_common.py``
  ``pa_decode_swa.py``  <- ``kernels/attention/pa_decode_swa.py``
  ``pa_decode_tile.py`` <- ``kernels/attention/pa_decode_tile.py``
  ``pa_support.py``     <- four host helpers from ``kernels/attention/pa_decode_fp8.py``

Only ``pa_decode_tile`` is a public entry point; see
``aiter.ops.flydsl.pa_decode``. ``pa_decode_swa`` is vendored for its
``compile_pa_decode_sw_reduce``, the split-KV combine used when
``num_partitions > 1``.
"""

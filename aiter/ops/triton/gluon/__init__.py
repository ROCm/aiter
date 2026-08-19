# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Deprecated import location for the gluon kernels.

The kernels now live in ``aiter/ops/triton/_gluon_kernels/<arch>/``. The modules
in this package stay behind as thin aliases so existing callers keep working
unchanged -- including out-of-tree ones, e.g. vLLM's MiniMax-M3 sparse PA imports
``aiter.ops.triton.gluon.pa_decode_gluon``. New code should import from
``_gluon_kernels/<arch>/`` (or better, from the wrapper in
``aiter/ops/triton/<category>/``) directly.

The triton>=3.6 check this package's ``__init__`` used to perform now lives in
``_gluon_kernels/_triton_version.py``; it is still invoked here so the old import
paths behave exactly as they did.
"""

from aiter.ops.triton._gluon_kernels._triton_version import require_gluon_triton

require_gluon_triton()

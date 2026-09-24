# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1201 SageAttention operators.

``torch.ops.aiter.gfx1201_sage_attention`` and
``torch.ops.aiter.gfx1201_norm_rope_attention`` are the public entry points.
HIP sources are built through ``compile_ops`` (``module_gfx1201_sage_attention``
and ``module_gfx1201_norm_rope_prepare``). The specialized code object lives in
``hsa/gfx1201/sage_attention/``.

The MiniMax-H3 pipelines stay on their submodules, for example
``aiter.ops.gfx1201.h3.H3TP2Workspace``, so importing this package does not
pull the fused FFN path.
"""

from .qk_norm import apply as qk_norm
from .rope import apply as rope
from .sage_attention import gfx1201_norm_rope_attention, gfx1201_sage_attention

__all__ = [
    "gfx1201_norm_rope_attention",
    "gfx1201_sage_attention",
    "qk_norm",
    "rope",
]

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deprecated alias for the relocated paged-MQA-logits gluon kernels.

Now specialized per arch under ``_gluon_kernels/<arch>/attention/pa_mqa_logits.py``;
this module forwards to the copy for the live arch. Note the gfx1250 copy carries
no ``_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx`` -- VarCtx has no
gfx1250 implementation. See this package's __init__.
"""

import sys

from aiter.jit.utils.chip_info import get_gfx

_arch = get_gfx()
if _arch == "gfx1250":
    from aiter.ops.triton._gluon_kernels.gfx1250.attention import (
        pa_mqa_logits as _impl,
    )
elif _arch == "gfx942":
    from aiter.ops.triton._gluon_kernels.gfx942.attention import pa_mqa_logits as _impl
else:
    from aiter.ops.triton._gluon_kernels.gfx950.attention import pa_mqa_logits as _impl

sys.modules[__name__] = _impl

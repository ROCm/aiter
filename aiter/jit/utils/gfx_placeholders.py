# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Values that mean "no architecture" in a tuned-CSV ``gfx`` cell.

Load-path backfill and FlyDSL AOT job resolution must use the same set so a
placeholder cannot be treated as a real ``FLYDSL_GPU_ARCH``. This module has
no GPU or rocminfo imports; both ``chip_info`` and AOT ``common`` share it.

``chip_info`` may be imported as a flat JIT util (``gfx_placeholders``) or as
``aiter.jit.utils.gfx_placeholders``. Register both names so the frozenset
and the CU map are the same objects either way.
"""

import sys

GFX_PLACEHOLDERS = frozenset(("", "0", "nan", "None"))

# Historical CU counts that predate the gfx column. Newer SKUs that share a
# count (gfx1250 also reports 256) or use a count this table never shipped
# without gfx (gfx1250 also reports 96) must write the real arch; they are
# never added here.
LEGACY_CU_NUM_TO_GFX = {
    80: "gfx942",
    304: "gfx942",
    256: "gfx950",
}

_this = sys.modules[__name__]
sys.modules.setdefault("gfx_placeholders", _this)
sys.modules.setdefault("aiter.jit.utils.gfx_placeholders", _this)

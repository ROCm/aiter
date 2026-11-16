# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The dispatch and combine stages, one module each, and which kind names
which. ``base`` says what a stage provides."""

from ..recipe import CombineKind, DispatchKind
from .combine_fused import _FusedScatterCombine
from .combine_mori import _MoriCombine
from .dispatch_compact import _CompactDispatch
from .dispatch_mori import _MoriDispatch

_DISPATCH_STAGES: dict = {
    DispatchKind.MORI: _MoriDispatch,
    DispatchKind.COMPACT: _CompactDispatch,
}
_COMBINE_STAGES: dict = {
    CombineKind.MORI: _MoriCombine,
    CombineKind.FUSED_SCATTER: _FusedScatterCombine,
}

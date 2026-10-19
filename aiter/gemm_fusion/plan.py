# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Plan interface implemented by each variant. No torch import at module level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class WindowReq:
    """What the runtime must allocate once per group (not per call)."""

    name: str
    bytes: int
    kind: str  # "p2p_window" | "control" | "sdma_queue"
    # Optional sharing tag: under the 'arena' policy every request with the same tag, from ANY owner, maps to ONE shared view of the arena.
    # `min_bytes` is the real requirement used for a shared window (defaults to `bytes`); `bytes` keeps the equal-size-per-plan convention
    # of the uniform policy. The uniform policy ignores both fields.
    share: Optional[str] = None
    min_bytes: Optional[int] = None


class Plan:
    n_stages: int = 0
    vid: str = ""

    def resources(self) -> List[WindowReq]:
        raise NotImplementedError

    def run(self, *, x, x_scale, w, w_scale, outs: Dict, meta: Dict, res, stream=None) -> None:
        """Run the covered stage prefix. ``res`` is a bound resources object
        (see aiter.gemm_fusion.resources.FusionResources)."""
        raise NotImplementedError

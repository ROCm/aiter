# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Plan interface implemented by each variant. No torch import at module level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class WindowReq:
    """What the runtime must allocate once per group (not per call)."""

    name: str
    bytes: int
    kind: str  # "p2p_window" | "control" | "sdma_queue"


class Plan:
    n_stages: int = 0
    vid: str = ""

    def resources(self) -> List[WindowReq]:
        raise NotImplementedError

    def run(self, *, x, x_scale, w, w_scale, outs: Dict, meta: Dict, res, stream=None) -> None:
        """Run the covered stage prefix. ``res`` is a bound resources object
        (see aiter.gemm_fusion.resources.FusionResources)."""
        raise NotImplementedError

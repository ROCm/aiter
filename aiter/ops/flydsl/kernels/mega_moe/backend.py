# SPDX-License-Identifier: Apache-2.0
"""Internal backend contract used by the MegaMoEV2 facade."""

from typing import Protocol

import torch


class MegaMoEBackend(Protocol):
    """Common public execution contract; transport internals remain backend-specific."""

    def forward(
        self,
        x_bf16: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        stream=None,
        slice_output: bool = True,
    ) -> torch.Tensor: ...

    def forward_prequant(
        self,
        x_q: torch.Tensor,
        scales: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        stream=None,
        slice_output: bool = True,
    ) -> torch.Tensor: ...

    def quantize(self, x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

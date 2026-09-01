# SPDX-License-Identifier: Apache-2.0
"""Compile-time problem shapes for gfx950 A8W4 GEMM2/TP kernels."""

from dataclasses import dataclass

SUPPORTED_TP_SIZES = (2, 4, 8)


@dataclass(frozen=True, slots=True)
class Gemm2TPShape:
    """Model dimensions specialized into one generated kernel."""

    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp_size: int = 8

    def __post_init__(self):
        if self.model_dim % 32:
            raise ValueError(
                "model_dim must be divisible by the 32-column MXFP8 scale "
                f"group, got {self.model_dim}"
            )
        if self.inter_dim % 128:
            raise ValueError(
                "inter_dim must be divisible by the 128-element shuffled "
                f"A8W4 K block, got {self.inter_dim}"
            )
        if self.tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(
                "gfx950 A8W4 GEMM2/TP currently requires TP in "
                f"{SUPPORTED_TP_SIZES}, "
                f"got {self.tp_size}"
            )

    @property
    def tag(self) -> str:
        return (
            f"h{self.model_dim}_i{self.inter_dim}_e{self.experts}"
            f"_k{self.topk}_tp{self.tp_size}"
        )

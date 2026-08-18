# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .activation import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    normalize_activation,
    validate_activation_parameters,
)


class TransportKind(str, Enum):
    COPY_STUB = "copy_stub"
    MORI_SHMEM = "mori_shmem"


@dataclass(frozen=True)
class KimiK3A4W4Shape:
    """Kimi-K3 routed-expert shape.

    ``inter_dim`` is TP-local.  The production EP16 topology owns 56 routed
    experts per rank; the EP8 single-host harness owns 112.
    """

    model_dim: int = 3584
    global_inter_dim: int = 3072
    tp_size: int = 8
    num_experts: int = 896
    topk: int = 16
    activation: str = "silu"
    swiglu_limit: float | None = None
    situ_beta: float = DEFAULT_SITUV2_BETA
    situ_linear_beta: float = DEFAULT_SITUV2_LINEAR_BETA

    def __post_init__(self) -> None:
        object.__setattr__(self, "activation", normalize_activation(self.activation))
        validate_activation_parameters(
            activation=self.activation,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )

    @property
    def inter_dim(self) -> int:
        if self.global_inter_dim % self.tp_size:
            raise ValueError("global_inter_dim must be divisible by tp_size")
        return self.global_inter_dim // self.tp_size


@dataclass(frozen=True)
class HierMegaMoETileConfig:
    """Static topology/workspace configuration shared by H1 and H2."""

    rank: int
    world_size: int
    logical_gpus_per_node: int
    num_experts: int = 896
    topk: int = 16
    model_dim: int = 3584
    inter_dim: int = 384
    activation: str = "silu"
    swiglu_limit: float | None = None
    situ_beta: float = DEFAULT_SITUV2_BETA
    situ_linear_beta: float = DEFAULT_SITUV2_LINEAR_BETA
    block_m: int = 32
    chunk_bytes: int = 64 * 1024
    ring_depth: int = 8
    num_qp_per_proxy: int = 4
    transport: TransportKind = TransportKind.COPY_STUB

    def __post_init__(self) -> None:
        object.__setattr__(self, "activation", normalize_activation(self.activation))
        validate_activation_parameters(
            activation=self.activation,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )
        if self.world_size <= 0 or self.logical_gpus_per_node <= 0:
            raise ValueError("world_size and logical_gpus_per_node must be positive")
        if self.world_size % self.logical_gpus_per_node:
            raise ValueError("world_size must be divisible by logical_gpus_per_node")
        if self.num_experts % self.world_size:
            raise ValueError("num_experts must be divisible by EP world_size")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank is outside the EP world")
        if self.topk <= 0 or self.topk > 64:
            raise ValueError("topk must be in [1, 64] for the packed source ABI")
        if self.block_m <= 0 or self.chunk_bytes <= 0 or self.ring_depth <= 0:
            raise ValueError("block_m/chunk_bytes/ring_depth must be positive")

    @property
    def num_logical_nodes(self) -> int:
        return self.world_size // self.logical_gpus_per_node

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.world_size

    @classmethod
    def production_ep16(cls, rank: int, **kwargs) -> "HierMegaMoETileConfig":
        return cls(rank=rank, world_size=16, logical_gpus_per_node=8, **kwargs)

    @classmethod
    def single_host_ep8_stub(cls, rank: int, **kwargs) -> "HierMegaMoETileConfig":
        # Two logical servers with four GPUs each, entirely inside one MI355X host.
        return cls(rank=rank, world_size=8, logical_gpus_per_node=4, **kwargs)

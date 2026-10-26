# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-shape inputs shared by the Opus Megatron local-expert benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronMoeBenchmarkCase:
    model_dim: int
    expert_inter_dim: int
    num_experts: int
    expert_parallel: int
    routes: int

    @property
    def local_experts(self) -> int:
        if self.num_experts % self.expert_parallel:
            raise ValueError("num_experts must be divisible by expert_parallel")
        return self.num_experts // self.expert_parallel


CASES = {
    # MBS=1, S=4096, TopK=2, EP=8.
    "mixtral-8x7b": MegatronMoeBenchmarkCase(4096, 14336, 8, 8, 1024),
    # MBS=2, S=4096, TopK=8, EP=8.
    "qwen3-235b-a22b": MegatronMoeBenchmarkCase(4096, 1536, 128, 8, 8192),
}

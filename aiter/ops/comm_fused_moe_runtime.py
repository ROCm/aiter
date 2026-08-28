# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lightweight runtime for communication-compute fused MoE Stage2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class CommFusedMoeRuntime:
    """Reuse ordinary MoE through Stage1, then run fused Stage2 + TP AR.

    Each prepared runner owns one exact token bucket and returns the complete
    replicated ``[M, H]`` output.
    """

    def __init__(
        self,
        *,
        runners: dict[int, Callable],
    ) -> None:
        self.runners = runners

    def supports(self, tokens: int) -> bool:
        from aiter.fused_moe import get_padded_M

        return int(get_padded_M(tokens)) in self.runners

    def run(
        self,
        *,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[], torch.Tensor] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
        **moe_args: Any,
    ) -> torch.Tensor:
        """Run ordinary MoE through Stage1 and fuse the complete Stage2 result."""

        from aiter.fused_moe import _fused_moe_impl, get_padded_M

        hidden_states = moe_args["hidden_states"]
        raw_tokens = int(hidden_states.shape[0])
        bucket = int(get_padded_M(raw_tokens))
        if bucket < raw_tokens:
            raise KeyError(f"no comm_fused bucket for {raw_tokens} tokens")
        runner = self.runners[bucket]

        if bucket != raw_tokens:
            topk_weight = moe_args["topk_weight"]
            topk_ids = moe_args["topk_ids"]
            padded_hidden = hidden_states.new_zeros((bucket, hidden_states.shape[1]))
            padded_weight = topk_weight.new_zeros((bucket, topk_weight.shape[1]))
            padded_ids = topk_ids.new_zeros((bucket, topk_ids.shape[1]))
            padded_hidden[:raw_tokens].copy_(hidden_states)
            padded_weight[:raw_tokens].copy_(topk_weight)
            padded_ids[:raw_tokens].copy_(topk_ids)
            moe_args["hidden_states"] = padded_hidden
            moe_args["topk_weight"] = padded_weight
            moe_args["topk_ids"] = padded_ids

        def stage2_override(**kwargs: Any):
            def launch():
                current_shared = shared_partial
                if before_stage2 is not None:
                    current_shared = before_stage2()
                if current_shared is None:
                    raise RuntimeError("comm-fused Stage2 requires shared_partial")
                if bucket != raw_tokens:
                    padded_shared = runner.output
                    padded_shared[:raw_tokens].copy_(current_shared)
                    padded_shared[raw_tokens:].zero_()
                    current_shared = padded_shared
                return runner(shared_partial=current_shared, **kwargs)

            if stage2_stream is None:
                return launch()
            with torch.cuda.stream(stage2_stream):
                return launch()

        output = _fused_moe_impl(
            **moe_args,
            _stage2_override=stage2_override,
        )
        return output if raw_tokens == bucket else output[:raw_tokens]


__all__ = ["CommFusedMoeRuntime"]

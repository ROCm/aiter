# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lightweight runtime for communication-compute fused MoE Stage2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch


class _RuntimeRunner(Protocol):
    output: torch.Tensor

    def output_rows_for(self, input_rows: int) -> int: ...

    def prepare_padded_shared_partial(
        self, shared_partial: torch.Tensor, input_rows: int
    ) -> torch.Tensor: ...

    def prepare_shared_partial(
        self, shared_partial: torch.Tensor
    ) -> torch.Tensor: ...

    def __call__(self, **kwargs: Any) -> torch.Tensor: ...


class _RunnerSet(Protocol):
    add_shared: bool

    def __contains__(self, tokens: int) -> bool: ...

    def __getitem__(self, tokens: int) -> _RuntimeRunner: ...


class CommFusedMoeRuntime:
    """Reuse ordinary MoE through Stage1, then run fused Stage2 + TP collective.

    Each prepared runner owns one exact token bucket. Runners with
    ``add_shared=True`` add a shared partial before TP reduction; the others
    return only the routed result in their declared output layout.

    Each runner owns its input-to-output row layout. Replicated collectives map
    one input row to one output row, while sharded collectives may return only
    a proportional subset. The runtime only consumes that layout contract and
    does not need to identify the runner's collective implementation.
    """

    def __init__(
        self,
        *,
        runners: _RunnerSet,
    ) -> None:
        self.runners = runners
        self.add_shared = runners.add_shared

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
        """Run ordinary MoE through Stage1 and fuse Stage2 with TP reduction."""

        from aiter.fused_moe import _fused_moe_impl, get_padded_M

        hidden_states = moe_args["hidden_states"]
        raw_tokens = int(hidden_states.shape[0])
        bucket = int(get_padded_M(raw_tokens))
        if bucket < raw_tokens:
            raise KeyError(f"no comm_fused bucket for {raw_tokens} tokens")
        runner = self.runners[bucket]
        output_rows = runner.output_rows_for(raw_tokens)

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
                add_shared = self.add_shared
                if add_shared and current_shared is None:
                    raise RuntimeError("comm-fused Stage2 requires shared_partial")
                if add_shared and bucket != raw_tokens:
                    current_shared = runner.prepare_padded_shared_partial(
                        current_shared, raw_tokens
                    )
                if add_shared:
                    current_shared = runner.prepare_shared_partial(current_shared)
                return runner(shared_partial=current_shared, **kwargs)

            if stage2_stream is None:
                return launch()
            caller_stream = torch.cuda.current_stream(hidden_states.device)
            if stage2_stream == caller_stream:
                return launch()
            stage2_stream.wait_stream(caller_stream)
            with torch.cuda.stream(stage2_stream):
                output = launch()
            caller_stream.wait_stream(stage2_stream)
            return output

        output = _fused_moe_impl(
            **moe_args,
            _stage2_override=stage2_override,
        )
        return output if output_rows == output.shape[0] else output[:output_rows]


__all__ = ["CommFusedMoeRuntime"]

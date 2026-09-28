# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lightweight runtime for communication-compute fused MoE Stage2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_BeforeStage2 = Callable[[int], torch.Tensor]
_BeforeSharedAdd = Callable[[], None]


class CommFusedMoeRuntime:
    """Reuse ordinary MoE through Stage1, then run fused Stage2 + TP collective.

    Each prepared runner owns one padded token bucket. Runners with
    ``add_shared=True`` add a shared partial before TP reduction; the others
    return only the routed result in their declared output layout.

    Each runner owns its input-to-output row layout. Replicated collectives map
    one input row to one output row, while sharded collectives may return only
    a proportional subset. The runtime only consumes that layout contract and
    does not need to identify the runner's collective implementation.
    """

    def __init__(self, *, runners) -> None:
        self.runners = runners

    def supports(self, tokens: int) -> bool:
        from aiter.fused_moe import get_padded_M

        bucket = int(get_padded_M(tokens))
        if bucket not in self.runners:
            return False
        try:
            self.runners.output_rows_for(bucket, tokens)
        except ValueError:
            return False
        return True

    def supports_ragged_m(self, _tokens: int) -> bool:
        """Current runners require rank-major DPA padding before all-gather."""

        return False

    def run(
        self,
        *,
        shared_partial: torch.Tensor | None,
        before_stage2: _BeforeStage2 | None = None,
        before_shared_add: _BeforeSharedAdd | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
        **moe_args: Any,
    ) -> torch.Tensor:
        """Run ordinary MoE through Stage1 and fuse Stage2 with TP reduction.

        ``before_stage2`` receives the runner's required shared-output row count
        so an asynchronous producer can perform any padding on its own stream.
        ``before_shared_add`` joins that producer only when the selected runner
        first consumes the shared output; no standalone ready kernel is used.
        """

        from aiter.fused_moe import _fused_moe_impl, get_padded_M

        hidden_states = moe_args["hidden_states"]
        raw_tokens = int(hidden_states.shape[0])
        bucket = int(get_padded_M(raw_tokens))
        if bucket < raw_tokens:
            raise KeyError(f"no comm_fused bucket for {raw_tokens} tokens")
        runner = self.runners[bucket]
        output_rows = self.runners.output_rows_for(bucket, raw_tokens)
        stage2_destination = runner.stage2_destination
        final_output = None
        if stage2_destination is not None:
            if moe_args.get("output") is not None:
                raise RuntimeError(
                    "comm-fused Stage2 output is incompatible with output="
                )
            moe_args["output"] = stage2_destination
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
                nonlocal final_output
                current_shared = shared_partial
                if before_stage2 is not None:
                    current_shared = before_stage2(runner.config.output_rows)
                add_shared = runner.config.shape.add_shared
                if add_shared and current_shared is None:
                    raise RuntimeError("comm-fused Stage2 requires shared_partial")
                if (
                    add_shared
                    and bucket != raw_tokens
                    and current_shared.shape[0] != runner.config.output_rows
                ):
                    current_shared = runner.prepare_padded_shared_partial(
                        current_shared, raw_tokens
                    )
                if add_shared:
                    current_shared = runner.prepare_shared_partial(current_shared)
                final_output = runner(
                    shared_partial=current_shared,
                    before_shared_add=before_shared_add,
                    **kwargs,
                )
                return (
                    stage2_destination
                    if stage2_destination is not None
                    else final_output
                )

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
        if final_output is not None:
            output = final_output
        return output if output_rows == output.shape[0] else output[:output_rows]


__all__ = ["CommFusedMoeRuntime"]

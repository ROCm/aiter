# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lightweight runtime for communication-compute fused MoE Stage2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_BeforeStage2 = Callable[[], torch.Tensor]
_BeforeStage2ForRows = Callable[[int], torch.Tensor]
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

    def bucket_for(self, tokens: int) -> int:
        """Return the standard padded bucket, extending past its upper bound."""

        from aiter.fused_moe import get_padded_M

        bucket = int(get_padded_M(tokens))
        if bucket < tokens:
            larger = [
                candidate for candidate in self.runners.configs if candidate >= tokens
            ]
            if larger:
                bucket = min(larger)
        return bucket

    def supports(self, tokens: int) -> bool:
        bucket = self.bucket_for(tokens)
        if bucket < tokens:
            return False
        if bucket not in self.runners:
            return False
        try:
            self.runners.output_rows_for(bucket, tokens)
        except ValueError:
            return False
        return True

    def supports_ragged_m(self, tokens: int) -> bool:
        """Return whether the selected runner accepts compact ragged RS rows."""

        bucket = self.bucket_for(tokens)
        if bucket < tokens or bucket not in self.runners:
            return False
        return self.runners[bucket].supports_ragged_m

    def run(
        self,
        *,
        shared_partial: torch.Tensor | None,
        before_stage2: _BeforeStage2 | None = None,
        before_stage2_for_rows: _BeforeStage2ForRows | None = None,
        before_shared_add: _BeforeSharedAdd | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
        reduce_scatter_sizes: list[int] | tuple[int, ...] | None = None,
        reuse_is_synchronized: bool = False,
        **moe_args: Any,
    ) -> torch.Tensor:
        """Run ordinary MoE through Stage1 and fuse Stage2 with TP reduction.

        ``before_stage2`` is the legacy zero-argument shared-output producer.
        ``before_stage2_for_rows`` receives the runner's required output row
        count. For compact ragged reduce-scatter this is the calling rank's
        real row count; fixed layouts retain their configured output capacity.
        Supplying both callbacks is an error.
        ``before_shared_add`` joins that producer only when the selected runner
        first consumes the shared output; no standalone ready kernel is used.
        ``reuse_is_synchronized`` lets an enclosing collective prove that all
        ranks finished the preceding Direct read before its Stage2 destination
        is overwritten. Other callers retain Direct's explicit deferred wait.
        """

        if before_stage2 is not None and before_stage2_for_rows is not None:
            raise ValueError(
                "before_stage2 and before_stage2_for_rows are mutually exclusive"
            )

        from aiter.fused_moe import _fused_moe_impl

        hidden_states = moe_args["hidden_states"]
        raw_tokens = int(hidden_states.shape[0])
        runner_kwargs = {}
        if reduce_scatter_sizes is None:
            bucket_input_rows = raw_tokens
        else:
            sizes = tuple(int(size) for size in reduce_scatter_sizes)
            if not sizes or any(size < 0 for size in sizes):
                raise ValueError(
                    "reduce_scatter_sizes must contain non-negative rank sizes"
                )
            if sum(sizes) != raw_tokens:
                raise ValueError(
                    "compact ragged input rows must equal sum(reduce_scatter_sizes): "
                    f"input={raw_tokens}, sizes={sizes}"
                )
            # Keep the compiled runner's fixed per-rank output capacity large
            # enough for the most-loaded rank. Input rows themselves remain
            # compact and any remaining bucket slack is appended only once at
            # the global tail below.
            bucket_input_rows = max(sizes) * len(sizes)
        bucket = self.bucket_for(bucket_input_rows)
        if bucket < bucket_input_rows:
            raise KeyError(
                f"no comm_fused bucket for required M={bucket_input_rows} "
                f"({raw_tokens} compact rows)"
            )
        runner = self.runners[bucket]
        if reuse_is_synchronized and runner.supports_external_reuse_sync:
            runner_kwargs["reuse_is_synchronized"] = True
        if reduce_scatter_sizes is None:
            output_rows = self.runners.output_rows_for(bucket, raw_tokens)
        else:
            if len(sizes) != runner.config.shape.tp_size:
                raise ValueError(
                    "reduce_scatter_sizes length must equal TP size: "
                    f"sizes={len(sizes)}, TP={runner.config.shape.tp_size}"
                )
            if not runner.supports_ragged_m:
                raise ValueError(f"{type(runner).__name__} does not support ragged M")
            local_rows = sizes[runner.rank]
            rank_offset = sum(sizes[: runner.rank])
            if local_rows > runner.config.output_rows:
                raise ValueError(
                    f"rank {runner.rank} has {local_rows} rows but runner "
                    f"capacity is {runner.config.output_rows}"
                )
            output_rows = local_rows
            runner_kwargs.update(
                local_rows=local_rows,
                rank_offset=rank_offset,
            )
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
                if before_stage2_for_rows is not None:
                    shared_rows = (
                        output_rows
                        if reduce_scatter_sizes is not None
                        else runner.config.output_rows
                    )
                    current_shared = before_stage2_for_rows(shared_rows)
                elif before_stage2 is not None:
                    current_shared = before_stage2()
                add_shared = runner.config.shape.add_shared
                if add_shared and current_shared is None:
                    raise RuntimeError("comm-fused Stage2 requires shared_partial")
                if (
                    add_shared
                    and reduce_scatter_sizes is None
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
                    **runner_kwargs,
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

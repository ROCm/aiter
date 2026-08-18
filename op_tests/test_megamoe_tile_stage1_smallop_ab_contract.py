# SPDX-License-Identifier: MIT
"""Static contract for the unfused-small-op versus fused Stage1 benchmark."""

from __future__ import annotations

import inspect

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_stage1_smallop_ab import (
    UNFUSED_FULL,
    UnfusedMoriStage1Path,
    _run_interleaved,
    _run_path,
)


def test_unfused_stage1_uses_existing_ops_and_stops_before_gmm2():
    source = inspect.getsource(UnfusedMoriStage1Path._run_components)
    assert "_input_quant_stage" in source
    assert "_dispatch_stage" in source
    assert "_prepare_stage" in source
    assert "_gmm1_stage" in source
    assert "_run_local_h2" not in source
    assert "_combine_stage" not in source
    assert UNFUSED_FULL in UnfusedMoriStage1Path.stage_names


def test_unfused_prepare_expands_duplicate_rank_routes():
    source = inspect.getsource(UnfusedMoriStage1Path._prepare_stage)
    assert "expand_local_routes=self.expand_local_routes" in source
    prime = inspect.getsource(UnfusedMoriStage1Path.prime_and_check)
    assert 'expected_topk = 2 if self.expand_local_routes else 1' in prime
    assert 'self._context["route_count"]' in prime
    assert "valid_routes != expected_routes" in prime


def test_mori_cleanup_is_after_the_complete_stage_event():
    iteration = inspect.getsource(UnfusedMoriStage1Path.run_iteration)
    cleanup = inspect.getsource(UnfusedMoriStage1Path.after_iteration)
    runner = inspect.getsource(_run_path)
    assert "timer.stage(UNFUSED_FULL" in iteration
    assert "combine" not in iteration
    assert "_cleanup_dispatch_lifecycle" in cleanup
    assert runner.index("timer.finish_iteration()") < runner.index(
        'getattr(path, "after_iteration", None)'
    )


def test_full_stage_contract_excludes_gmm2_and_combine():
    module_source = inspect.getsource(
        __import__(
            "op_tests.multigpu_tests.bench_megamoe_tile_ep16_stage1_smallop_ab",
            fromlist=["main"],
        ).main
    )
    assert '"gmm2_in_stage_event": False' in module_source
    assert '"combine_in_stage_event": False' in module_source
    assert '"same_bf16_start": True' in module_source
    assert '"same_gmm1_silu_a4_requant_end": True' in module_source


def test_interleaved_runner_swaps_order_outside_each_timed_event():
    source = inspect.getsource(_run_interleaved)
    assert "iteration % 2 == 0" in source
    assert "tuple(reversed(paths))" in source
    assert source.index("timer.finish_iteration()") < source.index(
        'getattr(path, "after_iteration", None)'
    )

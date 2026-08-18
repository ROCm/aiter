# SPDX-License-Identifier: MIT
"""CPU/static checks for the strict public two-kernel operator boundary."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import MegaMoETileA4W4
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import Stage2ArenaLayout


def test_public_constructor_and_forward_match_megamoe_v2_names():
    constructor = inspect.signature(MegaMoETileA4W4.__init__).parameters
    expected = {
        "rank",
        "world_size",
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "quant",
        "w1",
        "w1_scale",
        "w2",
        "w2_scale",
        "max_tok_per_rank",
        "mega_scheme",
        "swiglu_limit",
    }
    assert expected.issubset(constructor)
    assert constructor["stage1_transport"].default == "chunked"
    forward = inspect.signature(MegaMoETileA4W4.forward).parameters
    assert list(forward) == [
        "self",
        "x_bf16",
        "wts",
        "topk_ids",
        "stream",
        "slice_output",
    ]


def test_forward_source_has_exactly_two_launcher_calls_in_order():
    source = textwrap.dedent(inspect.getsource(MegaMoETileA4W4.forward))
    tree = ast.parse(source)
    launch_calls = []
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in ("_launch_stage1", "_launch_stage2"):
                launch_calls.append((node.lineno, function.attr))
            if function.attr in (
                "quantize",
                "zero_",
                "copy_",
                "index_copy_",
                "synchronize",
            ):
                forbidden.append(function.attr)
    assert [name for _, name in sorted(launch_calls)] == [
        "_launch_stage1",
        "_launch_stage2",
    ]
    assert forbidden == []


def test_composite_arena_is_one_aligned_nonoverlapping_window():
    stage1 = Stage1ArenaLayout.create()
    stage2 = Stage2ArenaLayout.create()
    composite = TwoKernelArenaLayout.compose(stage1, stage2)
    assert composite.stage2_offset % 4096 == 0
    assert composite.stage2_offset >= stage1.total_bytes
    assert composite.total_bytes >= composite.stage2_offset + stage2.total_bytes


def test_only_a4w4_quant_is_accepted_before_runtime_setup():
    kwargs = dict(
        rank=0,
        world_size=16,
        model_dim=7168,
        inter_dim=3072,
        experts=896,
        topk=16,
        max_tok_per_rank=128,
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
    )
    with pytest.raises(ValueError, match="quant='a4w4' only"):
        MegaMoETileA4W4._validate_static_contract(quant="a8w4", **kwargs)
    MegaMoETileA4W4._validate_static_contract(quant="a4w4", **kwargs)
    with pytest.raises(ValueError, match="stage1_transport"):
        MegaMoETileA4W4._validate_static_contract(
            quant="a4w4", stage1_transport="mori64x2", **kwargs
        )


def test_public_sparse_transport_selects_split_stage1_without_changing_stage2():
    source = inspect.getsource(MegaMoETileA4W4._compile_stage1)
    assert 'self.stage1_transport == "sparse_wqe"' in source
    assert "worker_blocks=self.stage1_worker_blocks" in source
    assert "diagnostic_split_fanout=sparse" in source
    assert "cco_geometry=self.stage1_transport" in source
    assert "tile_pipeline=sparse" in source
    assert "tile_pipeline_fanout_shards=16" in source
    validation = inspect.getsource(MegaMoETileA4W4._validate_launcher_contracts)
    assert '"gemm1_contraction": True' in validation
    assert '"full_stage1_fusion": True' in validation
    assert '"early_full_tile_enqueue": sparse' in validation
    assert '"concurrent_ready_queue_8_shards_256_all_roles_rejoin"' in validation
    stage2_source = inspect.getsource(MegaMoETileA4W4._compile_stage2)
    assert "WORK_SHARDS=8" in stage2_source
    assert "rank=self.rank" in stage2_source

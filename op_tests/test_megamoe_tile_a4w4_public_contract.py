# SPDX-License-Identifier: MIT
"""CPU/static checks for the strict public two-kernel operator boundary."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import MegaMoETileA4W4
from aiter.ops.flydsl.kernels.megamoe_tile.stage2 import (
    compile_megamoe_tile_ep16_stage2_a4w4,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import Stage2ArenaLayout
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    MoriFusedMoeBaselinePath,
)


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
        "max_routes_per_token_per_rank",
        "stage2_rail_quant_type",
        "stage2_gmm_work_swizzle",
        "stage2_window_n_groups",
        "stage2_ready_granularity",
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


def test_full_baseline_uses_production_mori_fused_moe_chain():
    source = textwrap.dedent(inspect.getsource(MoriFusedMoeBaselinePath._forward))
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in (
            "dispatch",
            "_fused_moe",
            "combine",
        ):
            calls.append((node.lineno, function.attr))
    assert [name for _, name in sorted(calls)] == [
        "dispatch",
        "_fused_moe",
        "combine",
    ]
    assert "self._quant_op(" in source
    assert "self._quant_q" in source
    assert "self._quant_scale" in source
    assert "a1_scale=recv_scales" in source
    assert "_prepare_local_a4w4" not in source
    assert "per_1x32_f4_quant" not in source


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


@pytest.mark.parametrize("max_tokens", [1, 128, 129, 4096])
def test_token_capacity_is_part_of_the_host_abi(max_tokens):
    stage1 = Stage1ArenaLayout.create(max_tokens=max_tokens)
    stage2 = Stage2ArenaLayout.create(max_tokens=max_tokens)
    assert stage1.max_tokens == max_tokens
    assert stage1.source_capacity == 16 * max_tokens
    assert stage2.max_tokens == max_tokens
    assert stage2.return_groups == (max_tokens + 3) // 4
    assert stage1.region("dispatch_staging").shape[1] == max_tokens
    assert stage2.region("node_dest_rank_mask").shape[2] == max_tokens


def test_compact_route_capacity_is_explicit_and_default_compatible():
    legacy = Stage1ArenaLayout.create(max_tokens=512)
    compact = Stage1ArenaLayout.create(
        max_tokens=512, max_routes_per_token_per_rank=1
    )
    assert legacy.max_routes_per_token_per_rank == legacy.topk
    assert legacy.route_capacity == legacy.source_capacity * legacy.topk
    assert compact.max_routes_per_token_per_rank == 1
    assert compact.route_capacity == compact.source_capacity
    assert compact.max_route_rows < legacy.max_route_rows
    stage2 = Stage2ArenaLayout.create(max_tokens=512)
    combined = TwoKernelArenaLayout.compose(compact, stage2)
    assert combined.total_bytes < 512 * 1024 * 1024


@pytest.mark.parametrize("route_cap", [0, 17])
def test_compact_route_capacity_must_be_within_topk(route_cap):
    with pytest.raises(ValueError, match=r"\[1, topk\]"):
        Stage1ArenaLayout.create(max_routes_per_token_per_rank=route_cap)


def test_stage2_rail_quant_layout_is_append_only_and_blockwise_128():
    plain = Stage2ArenaLayout.create(rail_quant_type="none")
    quant = Stage2ArenaLayout.create(rail_quant_type="fp8_blockwise")
    assert plain.rail_scale_dim == 0
    assert quant.rail_scale_dim == 7168 // 128 == 56
    assert plain.regions == quant.regions[: len(plain.regions)]
    assert quant.region("rail_fp8_tx_payload").offset >= plain.total_bytes
    assert (
        quant.region("rail_fp8_rx_payload").offset
        >= quant.region("rail_fp8_tx_payload").end
    )
    assert quant.region("rail_fp8_tx_payload").shape == (2, 128, 7168)
    assert quant.region("rail_fp8_rx_payload").shape == (2, 128, 7168)
    assert quant.region("rail_fp8_tx_payload").dtype is torch.uint8
    assert quant.region("rail_fp8_rx_payload").dtype is torch.uint8
    assert quant.region("rail_fp8_tx_scale").shape == (2, 128, 56)
    assert quant.region("rail_fp8_rx_scale").shape == (2, 128, 56)
    assert quant.region("rail_fp8_tx_scale").dtype is torch.float32
    assert quant.region("rail_fp8_rx_scale").dtype is torch.float32
    with pytest.raises(KeyError):
        plain.region("rail_fp8_tx_payload")


@pytest.mark.parametrize("quant_type", ["fp8", "fp4_blockwise"])
def test_stage2_rail_quant_rejects_unknown_modes(quant_type):
    with pytest.raises(ValueError, match="rail_quant_type"):
        Stage2ArenaLayout.create(rail_quant_type=quant_type)


@pytest.mark.parametrize("window_n_groups", [1, 2, 4, 7, 14])
def test_stage2_gmm_work_swizzle_contract(window_n_groups):
    kwargs = dict(
        rank=0,
        world_size=16,
        model_dim=7168,
        inter_dim=3072,
        experts=896,
        topk=16,
        quant="a4w4",
        max_tok_per_rank=128,
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
    )
    MegaMoETileA4W4._validate_static_contract(
        stage2_gmm_work_swizzle="n_major_window",
        stage2_window_n_groups=window_n_groups,
        **kwargs,
    )
    with pytest.raises(ValueError, match="stage2_gmm_work_swizzle"):
        MegaMoETileA4W4._validate_static_contract(
            stage2_gmm_work_swizzle="tile_queue",
            stage2_window_n_groups=window_n_groups,
            **kwargs,
        )


def test_stage2_gmm_work_swizzle_rejects_invalid_window():
    kwargs = dict(
        rank=0, world_size=16, model_dim=7168, inter_dim=3072,
        experts=896, topk=16, quant="a4w4", max_tok_per_rank=128,
        mega_scheme="hierarchical", swiglu_limit=0.0,
    )
    with pytest.raises(ValueError, match="stage2_window_n_groups"):
        MegaMoETileA4W4._validate_static_contract(
            stage2_gmm_work_swizzle="token_major",
            stage2_window_n_groups=3,
            **kwargs,
        )


def test_stage2_tile_ready_metadata_is_append_only_and_shape_derived():
    token = Stage2ArenaLayout.create(ready_granularity="token")
    tile = Stage2ArenaLayout.create(ready_granularity="tile")
    ready_groups = (7168 // 256 + 2 - 1) // 2
    assert tile.ready_group_tiles == 2
    assert tile.ready_group_count == ready_groups == 14
    assert token.regions == tile.regions[: len(token.regions)]
    assert tile.region("rank_tile_pending").shape == (2, 16 * 128, ready_groups)
    assert tile.region("rank_tile_ready").dtype is torch.int64
    assert tile.region("node_tile_arrived").shape == (2, 2 * 128, ready_groups)
    assert tile.region("node_tile_ready").dtype is torch.int64
    assert tile.region("node_ready_mask").shape == (2, 2 * 128)
    queue_capacity = 2 * 128 * ready_groups
    assert tile.region("tile_reduce_queue").shape == (2, queue_capacity)
    assert tile.region("tile_reduce_queue_ready").shape == (2, queue_capacity)
    assert tile.region("tile_reduce_queue_tail").shape == (2, 16)
    assert tile.region("tile_reduce_queue_head").shape == (2, 16)
    with pytest.raises(KeyError):
        token.region("rank_tile_pending")


def test_stage2_ready_granularity_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ready_granularity"):
        Stage2ArenaLayout.create(ready_granularity="n_group")


def test_stage2_ready_group_count_is_ceil_divided_and_mask_bounded():
    layout = Stage2ArenaLayout.create(
        hidden=1024, tile_n=256, ready_granularity="tile", ready_group_tiles=3
    )
    assert layout.hidden_tiles == 4
    assert layout.ready_group_count == 2
    with pytest.raises(ValueError, match="64-bit node_ready_mask"):
        Stage2ArenaLayout.create(
            hidden=7168,
            tile_n=64,
            ready_granularity="tile",
            ready_group_tiles=1,
        )


@pytest.mark.parametrize("max_tokens", [0, 4097])
def test_token_capacity_rejects_values_outside_supported_range(max_tokens):
    with pytest.raises(ValueError, match=r"\[1, 4096\]"):
        Stage1ArenaLayout.create(max_tokens=max_tokens)
    with pytest.raises(ValueError, match=r"\[1, 4096\]"):
        Stage2ArenaLayout.create(max_tokens=max_tokens)


def test_large_token_capacity_rejects_unscaled_experimental_protocols():
    kwargs = dict(
        rank=0,
        world_size=16,
        model_dim=7168,
        inter_dim=3072,
        experts=896,
        topk=16,
        quant="a4w4",
        max_tok_per_rank=129,
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
    )
    MegaMoETileA4W4._validate_static_contract(
        stage1_transport="chunked", **kwargs
    )
    with pytest.raises(ValueError, match="sparse_wqe"):
        MegaMoETileA4W4._validate_static_contract(
            stage1_transport="sparse_wqe", **kwargs
        )
    with pytest.raises(ValueError, match="staged_reduce"):
        Stage2ArenaLayout.create(
            max_tokens=129,
            include_rank_partials=True,
            include_staged_reduce=True,
        )
    with pytest.raises(ValueError, match="staged_ring"):
        Stage2ArenaLayout.create(
            max_tokens=129,
            include_rank_partials=True,
            include_staged_ring=True,
        )


def test_source_capacity_must_fit_packed_24_bit_identity(monkeypatch):
    import aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi as stage1_abi

    monkeypatch.setattr(stage1_abi, "MAX_FUSED_TOKENS_PER_RANK", 1 << 24)
    with pytest.raises(ValueError, match="24-bit packed source capacity"):
        Stage1ArenaLayout.create(max_tokens=1 << 20)


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
    assert 'accumulator_dtype="bf16"' in stage2_source
    assert "final_combine_blocks=14" in stage2_source
    assert 'gmm_schedule="persistent_queue"' in stage2_source
    assert '"stage2_return_chunk_tokens", 8' in stage2_source
    assert 'bf16_atomic_kind="buffer"' in stage2_source
    assert '"stage2_rail_return_schedule", "lockstep"' in stage2_source
    assert 'epilogue_schedule="lane32_meta"' in stage2_source
    assert '"stage2_n_tile_group", 2' in stage2_source
    assert 'group_pipeline_schedule="a_double_buffer"' in stage2_source
    assert "gmm_work_swizzle=self.stage2_gmm_work_swizzle" in stage2_source
    assert "window_n_groups=self.stage2_window_n_groups" in stage2_source

    stage2_parameters = inspect.signature(
        compile_megamoe_tile_ep16_stage2_a4w4
    ).parameters
    assert stage2_parameters["accumulator_dtype"].default == "bf16"
    assert stage2_parameters["return_chunk_tokens"].default == 8
    assert stage2_parameters["rail_return_schedule"].default == "lockstep"
    assert stage2_parameters["epilogue_schedule"].default == "lane32_meta"
    assert stage2_parameters["n_tile_group"].default == 2
    assert (
        stage2_parameters["group_pipeline_schedule"].default
        == "a_double_buffer"
    )

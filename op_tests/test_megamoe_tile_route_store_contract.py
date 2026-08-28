# SPDX-License-Identifier: MIT
"""Static contracts for the experimental Stage2 route-store reducer."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import (
    MegaMoETileA4W4,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    _permuted_arbitrary_destination_oracle,
    _permuted_arbitrary_topk_cpu,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_stage2_breakdown import (
    main as breakdown_main,
)
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    MegaMoETileA4W4CommProbe,
)
from op_tests.multigpu_tests.megamoe_tile_route_store_factory import (
    MegaMoETileA4W4RouteStore,
)
from op_tests.multigpu_tests import stress_megamoe_tile_ep16_sparse_routes
from op_tests.multigpu_tests import validate_megamoe_tile_route_store_ep16
from op_tests.multigpu_tests.validate_megamoe_tile_route_store_ep16 import (
    DIRECT_PACKED_FIXTURE_VERSION,
    REFERENCE_SCHEMA_VERSION,
    ROUTE_STORE_LAUNCHER_CONTRACT,
    _aggregate_generation_records,
    _compact_protocol_snapshot,
    _load_reference,
    _reference_path,
    _route_store_launcher_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_PATH = (
    REPO_ROOT / "aiter/ops/flydsl/kernels/megamoe_tile/stage2.py"
)


def _token_reducer_source() -> str:
    source = STAGE2_PATH.read_text()
    start = source.index(
        "# -------- Token-granular node reduction for route-store mode --------"
    )
    end = source.index("# ---------------- Source final-combine roles ----------------", start)
    return source[start:end]


def test_token_reducer_covers_every_bn_column_for_v4_and_v8():
    source = _token_reducer_source()

    # A 64-lane wave covers 128 BF16 values with a 4-byte vector and all 256
    # values with an 8-byte vector. Both the active-route reduction and the
    # zero-route clear must therefore use this compile-time column loop.
    assert "col_iterations = BN // (64 * reduce_elems)" in source
    assert source.count(
        "for col_iter in range_constexpr(col_iterations):"
    ) == 2
    assert source.count("fx.Int32(col_iter * 64 * reduce_elems)") == 2
    assert 256 // (64 * (4 // 2)) == 2
    assert 256 // (64 * (8 // 2)) == 1


def test_permuted_arbitrary_topk_cpu_fixture_is_deterministic_and_complete():
    shape = BenchmarkShape()
    ids, weights, rank_masks = _permuted_arbitrary_topk_cpu(shape, 0)
    repeat_ids, repeat_weights, repeat_masks = _permuted_arbitrary_topk_cpu(
        shape, 0
    )

    assert ids.device.type == "cpu"
    assert weights.device.type == "cpu"
    assert ids.shape == (128, 16)
    assert weights.shape == (128, 16)
    assert ids.equal(repeat_ids)
    assert weights.equal(repeat_weights)
    assert rank_masks == repeat_masks
    assert all(sum(mask.bit_count() for mask in row) == 16 for row in rank_masks)
    assert all(sum(mask != 0 for mask in row) == 6 for row in rank_masks)
    assert all(
        sum(
            mask.bit_count()
            for owner, mask in enumerate(row)
            if owner // shape.gpus_per_node == 0
        )
        == 8
        for row in rank_masks
    )

    oracle = _permuted_arbitrary_destination_oracle(shape, 0)
    assert oracle["routes"] == 2048
    assert oracle["unique_sources"] == 768
    assert sum(oracle["expert_count"]) == oracle["routes"]
    assert len(oracle["metadata_sha256"]) == 64


def test_shared_inputs_owns_the_arbitrary_fixture_and_full_packed_shapes():
    source = inspect.getsource(
        __import__(
            "op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path",
            fromlist=["_shared_inputs"],
        )._shared_inputs
    )
    assert 'route_pattern == "permuted-arbitrary-topk"' in source
    assert "_permuted_arbitrary_topk_cpu(" in source
    assert "(shape.local_experts, 2 * shape.inter, shape.hidden // 2)" in source
    assert "(shape.local_experts, shape.hidden, shape.inter // 2)" in source


def test_breakdown_plan_forwards_route_store_reducer_options(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_megamoe_tile_ep16_stage2_breakdown.py",
            "--path",
            "candidate",
            "--plan-only",
            "--candidate-node-accumulation-mode",
            "route_store",
            "--candidate-node-reduce-blocks",
            "16",
            "--candidate-node-reduce-vec-bytes",
            "8",
            "--candidate-node-reduce-schedule",
            "token",
            "--candidate-node-reduce-load-schedule",
            "interleaved",
        ],
    )

    assert breakdown_main() == 0
    line = capsys.readouterr().out.strip()
    prefix = "MEGAMOE_EP16_STAGE2_BREAKDOWN_PLAN "
    assert line.startswith(prefix)
    plan = json.loads(line[len(prefix) :])
    assert plan["candidate_node_accumulation_mode"] == "route_store"
    assert plan["candidate_node_reduce_blocks"] == 16
    assert plan["candidate_node_reduce_vec_bytes"] == 8
    assert plan["candidate_node_reduce_schedule"] == "token"
    assert plan["candidate_node_reduce_load_schedule"] == "interleaved"


def test_route_store_factory_defaults_and_launcher_manifest(monkeypatch):
    parameters = inspect.signature(MegaMoETileA4W4RouteStore.__init__).parameters
    assert parameters["stage2_node_reduce_blocks"].default == 16
    assert parameters["stage2_node_reduce_vec_bytes"].default == 8
    assert parameters["stage2_node_reduce_schedule"].default == "token"
    assert (
        parameters["stage2_node_reduce_load_schedule"].default
        == "interleaved"
    )

    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)
    operator = MegaMoETileA4W4RouteStore()
    assert operator.stage2_worker_blocks == 176
    expected = {
        "node_accumulation_mode": "route_store",
        "node_reduce_blocks": 16,
        "node_reduce_vec_bytes": 8,
        "node_reduce_schedule": "token",
        "node_reduce_load_schedule": "interleaved",
    }
    operator._stage2 = SimpleNamespace(
        **expected,
        architecture_contract=dict(expected),
    )
    monkeypatch.setattr(
        MegaMoETileA4W4,
        "_validate_launcher_contracts",
        lambda self: None,
    )
    operator._validate_launcher_contracts()

    operator._stage2.architecture_contract["node_reduce_vec_bytes"] = 4
    with pytest.raises(RuntimeError, match="launcher contract mismatch"):
        operator._validate_launcher_contracts()


def test_route_store_factory_rejects_vec16(monkeypatch):
    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)

    with pytest.raises(ValueError, match="route_store.*must be 4 or 8"):
        MegaMoETileA4W4RouteStore(stage2_node_reduce_vec_bytes=16)


def test_validation_requires_the_selected_route_store_launcher_manifest():
    expected = dict(ROUTE_STORE_LAUNCHER_CONTRACT)
    manifest = {
        name: value for name, value in expected.items() if name != "worker_blocks"
    }
    launcher = SimpleNamespace(**manifest, architecture_contract=dict(manifest))
    operator = SimpleNamespace(
        stage2_worker_blocks=expected["worker_blocks"],
        _stage2=launcher,
    )
    contract = _route_store_launcher_contract(operator)
    assert contract["launcher"] == expected
    assert contract["errors"] == []

    launcher.architecture_contract["node_reduce_load_schedule"] = "load_first"
    assert _route_store_launcher_contract(operator)["errors"]


def test_validation_accepts_an_explicit_vec4_route_store_launcher():
    expected = dict(ROUTE_STORE_LAUNCHER_CONTRACT)
    expected["node_reduce_vec_bytes"] = 4
    manifest = {
        name: value for name, value in expected.items() if name != "worker_blocks"
    }
    launcher = SimpleNamespace(**manifest, architecture_contract=dict(manifest))
    operator = SimpleNamespace(
        stage2_worker_blocks=expected["worker_blocks"],
        stage2_node_reduce_blocks=expected["node_reduce_blocks"],
        stage2_node_reduce_vec_bytes=4,
        stage2_node_reduce_schedule=expected["node_reduce_schedule"],
        stage2_node_reduce_load_schedule=expected[
            "node_reduce_load_schedule"
        ],
        _stage2=launcher,
    )

    contract = _route_store_launcher_contract(operator)

    assert contract["expected"] == expected
    assert contract["launcher"] == expected
    assert contract["errors"] == []


def test_comm_probe_grid_limit_depends_on_node_accumulation_mode(monkeypatch):
    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)

    direct = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=160,
        stage2_node_accumulation_mode="direct_atomic",
    )
    direct_min = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=16,
        stage2_node_accumulation_mode="direct_atomic",
    )
    direct_return_min = MegaMoETileA4W4CommProbe(
        stage2_mode="return_only",
        stage2_worker_blocks=15,
        stage2_node_accumulation_mode="direct_atomic",
    )
    route_store = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=176,
        stage2_node_accumulation_mode="route_store",
    )
    route_store_min = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=32,
        stage2_node_accumulation_mode="route_store",
    )
    route_store_return_min = MegaMoETileA4W4CommProbe(
        stage2_mode="return_only",
        stage2_worker_blocks=31,
        stage2_node_accumulation_mode="route_store",
    )
    route_store_r32 = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=192,
        stage2_node_accumulation_mode="route_store",
        stage2_node_reduce_blocks=32,
    )
    assert direct.stage2_worker_blocks == 160
    assert direct_min.stage2_worker_blocks == 16
    assert direct_return_min.stage2_worker_blocks == 15
    assert route_store.stage2_worker_blocks == 176
    assert route_store_min.stage2_worker_blocks == 32
    assert route_store_return_min.stage2_worker_blocks == 31
    assert route_store_r32.stage2_worker_blocks == 192

    with pytest.raises(ValueError, match=r"\[16, 160\].*direct_atomic/full"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=15,
            stage2_node_accumulation_mode="direct_atomic",
        )
    with pytest.raises(
        ValueError, match=r"\[15, 160\].*direct_atomic/return_only"
    ):
        MegaMoETileA4W4CommProbe(
            stage2_mode="return_only",
            stage2_worker_blocks=14,
            stage2_node_accumulation_mode="direct_atomic",
        )
    with pytest.raises(ValueError, match=r"\[16, 160\].*direct_atomic/full"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=161,
            stage2_node_accumulation_mode="direct_atomic",
        )
    with pytest.raises(ValueError, match=r"\[32, 176\].*route_store/full"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=31,
            stage2_node_accumulation_mode="route_store",
        )
    with pytest.raises(
        ValueError, match=r"\[31, 176\].*route_store/return_only"
    ):
        MegaMoETileA4W4CommProbe(
            stage2_mode="return_only",
            stage2_worker_blocks=30,
            stage2_node_accumulation_mode="route_store",
        )
    with pytest.raises(ValueError, match=r"\[32, 176\].*route_store/full"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=177,
            stage2_node_accumulation_mode="route_store",
        )
    with pytest.raises(ValueError, match=r"\[48, 192\].*route_store/full"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=193,
            stage2_node_accumulation_mode="route_store",
            stage2_node_reduce_blocks=32,
        )


def test_comm_probe_load_schedule_is_validated_and_forwarded(monkeypatch):
    parameters = inspect.signature(MegaMoETileA4W4CommProbe.__init__).parameters
    assert parameters["stage2_node_reduce_load_schedule"].default == "interleaved"

    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)
    load_first = MegaMoETileA4W4CommProbe(
        stage2_node_reduce_load_schedule="load_first"
    )
    assert load_first.stage2_node_reduce_load_schedule == "load_first"
    with pytest.raises(ValueError, match="node-reduce load schedule"):
        MegaMoETileA4W4CommProbe(
            stage2_node_reduce_load_schedule="unsupported"
        )

    compile_source = inspect.getsource(
        MegaMoETileA4W4CommProbe._build_stage2_probe_launcher
    )
    assert "node_reduce_load_schedule=(" in compile_source
    assert "self.stage2_node_reduce_load_schedule" in compile_source


def test_comm_probe_route_store_debug_helpers_have_bound_addresses_and_counts():
    direct_source = inspect.getsource(
        MegaMoETileA4W4CommProbe.debug_direct_tile_snapshot
    )
    assert 's2_ptr("node_partial_done")' in direct_source
    assert (
        'read_window_u32(\n                        ptr("node_partial_done")'
        not in direct_source
    )
    assert 's2_ptr("node_partial_ready")' in direct_source
    assert 'in ("route_store", "rank_local")' in direct_source
    assert "ready.append(int(node_ready_all[token_index] >= generation))" in direct_source
    assert "for value in node_ready_all" in direct_source

    scoreboard_source = inspect.getsource(
        MegaMoETileA4W4CommProbe.debug_stage2_scoreboard_snapshot
    )
    assert "partial_done_all = (" in scoreboard_source
    assert 'ptr("node_partial_done")' in scoreboard_source
    assert 'self.stage2_node_reduce_schedule == "tile"' in scoreboard_source
    assert scoreboard_source.index("partial_done_all = (") < scoreboard_source.index(
        '"node_partial_done_mismatch"'
    )


@pytest.mark.parametrize("flag", ("--route-store-stage2", "--poison-stage2"))
def test_stage2_stress_only_flags_require_expected_sweep(
    flag, monkeypatch, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        ["stress_megamoe_tile_ep16_sparse_routes.py", flag],
    )
    with pytest.raises(SystemExit) as error:
        stress_megamoe_tile_ep16_sparse_routes.main()
    assert error.value.code == 2
    assert "require --stage2-expected-sweep" in capsys.readouterr().err


def test_stage2_stress_pins_route_store_load_schedule():
    source = inspect.getsource(stress_megamoe_tile_ep16_sparse_routes.main)
    assert "args.stage2_node_reduce_load_schedule" in source
    assert 'else "interleaved"' in source
    assert (
        "stage2_node_reduce_vec_bytes=args.stage2_node_reduce_vec_bytes"
        in source
    )


def _fake_protocol_snapshot(*, generation: int, mismatch: int = 0):
    return {
        "generation": generation,
        "stage1_done": generation,
        "comm_role_eos": [generation] * 8,
        "node_atomic_expected": [8] * 128,
        "node_atomic_done": [8] * 128,
        "node_atomic_ready": [1] * 128,
        "protocol_error_count": [0],
        "stage1_error_count": 0,
        "stage2_error_count": 0,
        "node_expected_uniform_mismatch": 0,
        "node_expected_done_mismatch": mismatch,
        "node_not_ready": 0,
        "node_route_store_not_ready": 0,
        "node_token_done_mismatch": 0,
        "node_partial_done_mismatch": 0,
        "queue_permutation_mismatch": 0,
    }


def test_fake_protocol_aggregate_keeps_every_rank_and_uses_inclusive_threshold():
    records = []
    for rank in range(2):
        protocol = _compact_protocol_snapshot(
            _fake_protocol_snapshot(generation=7),
            rank=rank,
            generation=7,
        )
        records.append(
            {
                "rank": rank,
                "generation_index": 3,
                "device_generation": 7,
                "protocol": protocol,
                "comparisons": [
                    {
                        "rank": rank,
                        "label": "route_vs_direct",
                        "rel_l2": 0.049,
                        "max_abs": 0.25 + rank,
                    }
                ],
                "errors": [],
            }
        )
    summary, failures = _aggregate_generation_records(
        records, rel_l2_threshold=0.05
    )
    assert failures == []
    assert summary["generation_index"] == 3
    assert summary["device_generation"] == 7
    assert len(summary["ranks"]) == 2
    assert summary["rank_max_rel_l2"] == 0.049
    assert summary["rank_max_abs"] == 1.25
    assert summary["protocol_error_count"] == 0

    records[1]["comparisons"][0]["rel_l2"] = 0.05
    records[1]["protocol"] = _compact_protocol_snapshot(
        _fake_protocol_snapshot(generation=7, mismatch=1),
        rank=1,
        generation=7,
    )
    _, failures = _aggregate_generation_records(
        records, rel_l2_threshold=0.05
    )
    assert any("rel_l2=0.05 >= 0.05" in failure for failure in failures)
    assert any("node_expected_done_mismatch=1" in failure for failure in failures)

    stale = _fake_protocol_snapshot(generation=7)
    stale["stage1_done"] = 6
    stale["comm_role_eos"] = [7] * 7 + [6]
    compact = _compact_protocol_snapshot(stale, rank=0, generation=7)
    assert any("stage1_done=6" in error for error in compact["errors"])
    assert any("comm_role_eos=" in error for error in compact["errors"])

    run_source = inspect.getsource(validate_megamoe_tile_route_store_ep16._run)
    assert "_validate_direct_tile_debug_snapshot(" in run_source
    assert 'protocol["errors"].append(' in run_source
    assert run_source.index("operator.debug_direct_tile_snapshot()") < run_source.index(
        "dist.all_gather_object(gathered, local_record)"
    )


def test_reference_file_is_fixture_scoped_and_preserves_every_generation(
    tmp_path,
):
    shape = SimpleNamespace(tokens=2, hidden=2)
    metadata = {
        "route_pattern": "permuted-arbitrary-topk",
        "rank": 3,
        "packed_fixture_version": DIRECT_PACKED_FIXTURE_VERSION,
    }
    path = _reference_path(tmp_path, metadata["route_pattern"], 3)
    assert path != _reference_path(tmp_path, "paired-rank-half-remote", 3)
    outputs = [
        torch.full((2, 2), index, dtype=torch.bfloat16)
        for index in range(4)
    ]
    torch.save(
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "metadata": metadata,
            "generations": 4,
            "device_generations": [1, 2, 3, 4],
            "outputs": outputs,
        },
        path,
    )

    loaded, device_generations, errors = _load_reference(
        path, shape=shape, metadata=metadata, generations=4
    )
    assert errors == []
    assert loaded is not None
    assert len(loaded) == 4
    assert device_generations == [1, 2, 3, 4]
    assert all(left.equal(right) for left, right in zip(loaded, outputs))

    _, _, errors = _load_reference(
        path,
        shape=shape,
        metadata={
            "route_pattern": "paired-rank-half-remote",
            "rank": 3,
            "packed_fixture_version": DIRECT_PACKED_FIXTURE_VERSION,
        },
        generations=4,
    )
    assert "reference fixture metadata does not match this run" in errors

    torch.save(
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "metadata": metadata,
            "generations": 4,
            "device_generations": [1, 2, 4, 5],
            "outputs": outputs,
        },
        path,
    )
    _, _, errors = _load_reference(
        path, shape=shape, metadata=metadata, generations=4
    )
    assert any("not consecutive" in error for error in errors)


@pytest.mark.parametrize(
    "schema_version,output,match",
    [
        (1, torch.zeros((2, 2), dtype=torch.bfloat16), "schema_version=1"),
        (
            REFERENCE_SCHEMA_VERSION,
            torch.zeros((1, 2), dtype=torch.bfloat16),
            "shape=(1, 2)",
        ),
        (
            REFERENCE_SCHEMA_VERSION,
            torch.zeros((2, 2), dtype=torch.float32),
            "dtype=torch.float32",
        ),
    ],
)
def test_reference_rejects_old_schema_wrong_shape_or_wrong_dtype(
    tmp_path, schema_version, output, match
):
    shape = SimpleNamespace(tokens=2, hidden=2)
    metadata = {
        "route_pattern": "permuted-arbitrary-topk",
        "rank": 0,
        "packed_fixture_version": DIRECT_PACKED_FIXTURE_VERSION,
    }
    path = _reference_path(tmp_path, metadata["route_pattern"], 0)
    torch.save(
        {
            "schema_version": schema_version,
            "metadata": metadata,
            "generations": 4,
            "device_generations": [1, 2, 3, 4],
            "outputs": [output.clone() for _ in range(4)],
        },
        path,
    )

    _, _, errors = _load_reference(
        path, shape=shape, metadata=metadata, generations=4
    )
    assert any(match in error for error in errors)


def test_validation_rejects_fewer_than_four_generations_before_dist_setup(
    monkeypatch, tmp_path
):
    source = inspect.getsource(validate_megamoe_tile_route_store_ep16.main)
    assert 'parser.add_argument("--generations", type=int, default=4)' in source
    assert 'default="paired-rank-half-remote"' in source
    assert 'direct_packed_weights=True' in source
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_megamoe_tile_route_store_ep16.py",
            "--mode",
            "direct",
            "--output-dir",
            str(tmp_path),
            "--generations",
            "3",
        ],
    )
    with pytest.raises(SystemExit) as error:
        validate_megamoe_tile_route_store_ep16.main()
    assert error.value.code == 2

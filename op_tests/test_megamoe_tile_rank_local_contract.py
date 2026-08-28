# SPDX-License-Identifier: MIT
"""Static wiring contracts for experimental rank-local Stage2 accumulation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import (
    MegaMoETileA4W4,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    DIRECT_TILE_STAGE1_CONTRACT,
    RANK_LOCAL_STAGE2_CONTRACT,
    _validate_direct_tile_operator,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_stage2_breakdown import (
    main as breakdown_main,
)
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    MegaMoETileA4W4CommProbe,
)
from op_tests.multigpu_tests.megamoe_tile_rank_local_factory import (
    MegaMoETileA4W4RankLocal,
)
from op_tests.multigpu_tests.stress_megamoe_tile_ep16_sparse_routes import (
    main as stress_main,
)
from op_tests.multigpu_tests.validate_megamoe_tile_route_store_ep16 import (
    RANK_LOCAL_LAUNCHER_CONTRACT,
    _rank_local_launcher_contract,
    main as validation_main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_PATH = (
    REPO_ROOT / "aiter/ops/flydsl/kernels/megamoe_tile/stage2.py"
)


def _rank_local_reducer_source() -> str:
    source = STAGE2_PATH.read_text()
    start = source.index("        def reduce_rank_queue_slot(")
    end = source.index("        def consume_dynamic_rank_reduce(", start)
    return source[start:end]


def test_rank_local_stage2_is_local_atomic_then_peer_pull():
    source = STAGE2_PATH.read_text()

    assert 'node_accumulation_mode == "rank_local"' in source
    assert "rank_accumulator_ptr" in source
    assert "rank_pending_ptr" in source
    assert "publish_rank_group" in source
    assert "rank_ready_ptr" in source
    assert "rank_reduce_queue_ready_ptr" in source
    assert "rank_reduce_queue_head_ptr" in source
    assert "active_count = fx.Int32(0)" in source
    assert "reduce_work_items" in source
    assert "def reduce_rank_queue_slot(" in source
    assert "def consume_dynamic_rank_reduce(" in source
    assert 'node_reduce_work_schedule == "static_strided"' in source
    assert "if const_expr(node_reduce_rejoin_blocks > 0)" in source
    assert "queued < reduce_work_items" in source
    assert "gmm_worker < fx.Int32(node_reduce_rejoin_blocks)" in source
    assert "queue-producing, reducer-enabled" in source
    assert 'rail_return_schedule == "compact"' in source
    assert "rank_tx_slot_ptr" in source
    assert "rank_rx_slot_ptr" in source
    assert "peer_rank_rsrc" in source
    assert "rank_mask_rsrc" in source
    assert "rank_local_atomic_then_peer_pull_node_reduce" in source
    assert "source * fx.Int32(HIDDEN)" in source


def test_rank_local_vec16_uses_one_masked_half_wave_per_tile():
    full_source = STAGE2_PATH.read_text()
    source = _rank_local_reducer_source()

    assert "node_reduce_vec_bytes not in (4, 8, 16)" in full_source
    assert 'node_accumulation_mode != "rank_local"' in full_source
    assert "16-byte node reduction requires rank_local" in full_source
    assert "if const_expr(node_reduce_vec_bytes == 16):" in source
    assert "col_iterations = BN // (32 * reduce_elems)" in source
    assert "lane_active = lane < fx.Int32(32)" in source
    assert "col_in_tile = lane_active.select(" in source
    assert "fx.Int32(col_iter * 32 * reduce_elems)" in source
    assert source.count("mask=peer_active & lane_active") == 2
    assert source.count("vec_width=reduce_i32s") == 4
    assert "mask=lane_active" in source
    assert "mask=(rank_mask != fx.Int32(0)) & lane_active" in source

    # The established vec4/vec8 lane geometry remains in the compile-time
    # else path, rather than inheriting the half-wave mask.
    assert "col_iterations = BN // (64 * reduce_elems)" in source
    assert "fx.Int32(col_iter * 64 * reduce_elems)" in source
    assert source.count("mask=peer_active,") == 2


def test_rank_local_factory_defaults_and_launcher_manifest(monkeypatch):
    parameters = inspect.signature(MegaMoETileA4W4RankLocal.__init__).parameters
    assert parameters["stage2_node_reduce_blocks"].default == 16
    assert parameters["stage2_node_reduce_vec_bytes"].default == 8
    assert parameters["stage2_node_reduce_schedule"].default == "token"
    assert (
        parameters["stage2_node_reduce_load_schedule"].default
        == "interleaved"
    )
    assert (
        parameters["stage2_node_reduce_work_schedule"].default
        == "static_strided"
    )
    assert parameters["stage2_node_reduce_rejoin_blocks"].default == 0
    assert (
        parameters["stage2_rank_epilogue_lds_addressing"].default
        == "expanded"
    )
    probe_parameters = inspect.signature(
        MegaMoETileA4W4CommProbe.__init__
    ).parameters
    assert probe_parameters["stage2_return_chunk_tokens"].default == 8
    assert probe_parameters["stage2_rail_return_schedule"].default == "lockstep"
    assert parameters["stage2_return_chunk_tokens"].default == 16
    assert parameters["stage2_rail_return_schedule"].default == "compact"

    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)
    operator = MegaMoETileA4W4RankLocal()
    assert operator.stage2_node_accumulation_mode == "rank_local"
    assert operator.stage2_worker_blocks == 176

    expected = {
        "node_accumulation_mode": "rank_local",
        "node_reduce_blocks": 16,
        "node_reduce_vec_bytes": 8,
        "node_reduce_schedule": "token",
        "node_reduce_load_schedule": "interleaved",
        "node_reduce_work_schedule": "static_strided",
        "node_reduce_rejoin_blocks": 0,
        "rank_epilogue_lds_addressing": "expanded",
        "rank_accumulation_mode": "atomic",
        "return_chunk_tokens": 16,
        "rail_return_schedule": "compact",
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

    operator._stage2.architecture_contract["node_accumulation_mode"] = (
        "direct_atomic"
    )
    with pytest.raises(RuntimeError, match="rank-local Stage2 launcher"):
        operator._validate_launcher_contracts()


def test_rank_local_factory_accepts_vec16_and_preserves_vec8_default(
    monkeypatch,
):
    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)

    operator = MegaMoETileA4W4RankLocal(stage2_node_reduce_vec_bytes=16)
    load_first = MegaMoETileA4W4RankLocal(
        stage2_node_reduce_vec_bytes=16,
        stage2_node_reduce_load_schedule="load_first",
    )

    assert operator.stage2_node_reduce_vec_bytes == 16
    assert load_first.stage2_node_reduce_load_schedule == "load_first"
    assert (
        inspect.signature(MegaMoETileA4W4RankLocal.__init__)
        .parameters["stage2_node_reduce_vec_bytes"]
        .default
        == 8
    )
    with pytest.raises(ValueError, match="must be 4, 8, or 16"):
        MegaMoETileA4W4RankLocal(stage2_node_reduce_vec_bytes=32)


def test_rank_local_factory_accepts_dynamic_rejoin_and_rejects_static_rejoin(
    monkeypatch,
):
    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)

    operator = MegaMoETileA4W4RankLocal(
        stage2_node_reduce_work_schedule="dynamic_head",
        stage2_node_reduce_rejoin_blocks=8,
    )
    assert operator.stage2_node_reduce_work_schedule == "dynamic_head"
    assert operator.stage2_node_reduce_rejoin_blocks == 8

    with pytest.raises(ValueError, match="dynamic_head reduction"):
        MegaMoETileA4W4RankLocal(stage2_node_reduce_rejoin_blocks=8)


def test_public_operator_selects_rank_local_abi_without_changing_default():
    source = inspect.getsource(MegaMoETileA4W4.__init__)

    assert 'stage2_node_accumulation_mode == "route_store"' in source
    assert 'stage2_node_accumulation_mode == "rank_local"' in source
    assert '"stage2_node_accumulation_mode", "direct_atomic"' in source
    assert "include_rank_partials=" in source


def test_debug_deferred_ready_covers_both_optional_accumulation_modes():
    snapshot_source = inspect.getsource(
        MegaMoETileA4W4.debug_direct_tile_snapshot
    )
    timeline_source = inspect.getsource(MegaMoETileA4W4.debug_device_timeline)

    assert "self.stage2_layout.include_rank_partials" in snapshot_source
    assert '("route_store", "rank_local")' in timeline_source


def test_compile_smoke_exposes_rank_local_without_changing_default():
    source = (
        REPO_ROOT / "scripts/megamoe_tile/compile_ep16_stage2_fused.py"
    ).read_text()

    assert 'choices=("direct_atomic", "route_store", "rank_local")' in source
    assert 'default="direct_atomic"' in source
    assert (
        'include_rank_partials=args.node_accumulation_mode == "rank_local"'
        in source
    )
    assert '"--node-reduce-work-schedule"' in source
    assert 'choices=("static_strided", "dynamic_head")' in source
    assert '"--node-reduce-rejoin-blocks"' in source
    assert "choices=(0, 8, 16, 32)" in source
    assert 'choices=(4, 8, 16), default=4' in source
    assert "16-byte node reduction requires" in source


def test_two_kernel_validator_accepts_rank_local_manifest():
    reducer = {
        "node_reduce_blocks": 16,
        "node_reduce_vec_bytes": 8,
        "node_reduce_schedule": "token",
        "node_reduce_load_schedule": "interleaved",
        "node_reduce_work_schedule": "static_strided",
        "node_reduce_rejoin_blocks": 0,
    }
    operator = SimpleNamespace(
        _stage1=SimpleNamespace(
            gemm1_contraction=True,
            architecture_contract=dict(DIRECT_TILE_STAGE1_CONTRACT),
        ),
        _stage2=SimpleNamespace(
            gemm2_contraction=True,
            kernel_name="megamoe_tile_ep16_stage2_rank_local",
            node_accumulation_mode="rank_local",
            architecture_contract={**RANK_LOCAL_STAGE2_CONTRACT, **reducer},
            **reducer,
        ),
    )

    _validate_direct_tile_operator(operator)


def test_validation_accepts_rank_local_launcher_manifest():
    expected = dict(RANK_LOCAL_LAUNCHER_CONTRACT)
    manifest = {
        name: value for name, value in expected.items() if name != "worker_blocks"
    }
    operator = SimpleNamespace(
        stage2_worker_blocks=expected["worker_blocks"],
        stage2_node_reduce_vec_bytes=expected["node_reduce_vec_bytes"],
        _stage2=SimpleNamespace(
            **manifest,
            architecture_contract=dict(manifest),
        ),
    )

    assert _rank_local_launcher_contract(operator)["errors"] == []


def test_validation_accepts_dynamic_rank_local_launcher_manifest():
    expected = {
        **RANK_LOCAL_LAUNCHER_CONTRACT,
        "node_reduce_load_schedule": "load_first",
        "node_reduce_work_schedule": "dynamic_head",
        "node_reduce_rejoin_blocks": 8,
    }
    manifest = {
        name: value for name, value in expected.items() if name != "worker_blocks"
    }
    operator = SimpleNamespace(
        stage2_worker_blocks=expected["worker_blocks"],
        stage2_node_reduce_vec_bytes=expected["node_reduce_vec_bytes"],
        stage2_node_reduce_load_schedule="load_first",
        stage2_node_reduce_work_schedule="dynamic_head",
        stage2_node_reduce_rejoin_blocks=8,
        _stage2=SimpleNamespace(
            **manifest,
            architecture_contract=dict(manifest),
        ),
    )

    assert _rank_local_launcher_contract(operator)["errors"] == []


def test_comm_probe_allocates_rank_local_reducer_roles(monkeypatch):
    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)

    operator = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=176,
        stage2_node_accumulation_mode="rank_local",
        stage2_node_reduce_blocks=16,
    )
    assert operator.stage2_worker_blocks == 176
    assert operator.stage2_node_accumulation_mode == "rank_local"

    with pytest.raises(ValueError, match=r"\[32, 176\].*rank_local/full"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=177,
            stage2_node_accumulation_mode="rank_local",
            stage2_node_reduce_blocks=16,
        )


def test_comm_probe_allows_vec16_only_for_rank_local(monkeypatch):
    monkeypatch.setattr(MegaMoETileA4W4, "__init__", lambda self, *a, **k: None)

    operator = MegaMoETileA4W4CommProbe(
        stage2_worker_blocks=176,
        stage2_node_accumulation_mode="rank_local",
        stage2_node_reduce_blocks=16,
        stage2_node_reduce_vec_bytes=16,
    )
    assert operator.stage2_node_reduce_vec_bytes == 16

    with pytest.raises(ValueError, match="requires rank_local"):
        MegaMoETileA4W4CommProbe(
            stage2_worker_blocks=176,
            stage2_node_accumulation_mode="route_store",
            stage2_node_reduce_blocks=16,
            stage2_node_reduce_vec_bytes=16,
        )


def test_comm_probe_rank_local_snapshots_use_logical_completion_state():
    helper_source = inspect.getsource(
        MegaMoETileA4W4CommProbe._debug_rank_local_completion
    )
    direct_source = inspect.getsource(
        MegaMoETileA4W4CommProbe.debug_direct_tile_snapshot
    )
    scoreboard_source = inspect.getsource(
        MegaMoETileA4W4CommProbe.debug_stage2_scoreboard_snapshot
    )
    zero_source = inspect.getsource(
        MegaMoETileA4W4CommProbe.debug_stage2_zero_payload_snapshot
    )

    for field in (
        "rank_local_active_tokens",
        "rank_local_pending_nonzero",
        "rank_local_pending_nonzero_all",
        "rank_local_ready_missing",
        "rank_local_ready_unexpected",
    ):
        assert field in helper_source
    assert "pending_raw[index * 16]" in helper_source
    assert "source not in active_sources" in helper_source
    assert "**rank_local_state" in direct_source
    assert "**rank_local_state" in scoreboard_source
    assert 'self.stage2_node_accumulation_mode == "rank_local"' in direct_source
    assert 'self.stage2_node_accumulation_mode == "rank_local"' in scoreboard_source
    assert "else min(done_all[start:end])" in direct_source
    assert "else min(int(value) for value in done_slice)" in scoreboard_source
    assert direct_source.count("if not rank_local") >= 3
    assert scoreboard_source.count("if not rank_local") >= 3
    assert 's2_ptr("node_partial_ready")' in direct_source
    assert 'ptr("node_partial_ready")' in scoreboard_source
    assert '"final_expected"' in direct_source
    assert '"rank_accumulator_nonzero_bytes"' in zero_source
    assert '"rank_token_pending_nonzero"' in zero_source
    assert '"rank_token_ready_missing"' in zero_source
    assert '"rank_token_ready_unexpected"' in zero_source


def test_rank_local_completion_summary_uses_only_logical_pending_lanes(
    monkeypatch,
):
    from aiter.ops.flydsl.kernels.megamoe_tile import cco

    source_capacity = 4
    pending = [0x7F7F7F7F] * (source_capacity * 16)
    for source, value in enumerate((0, 9, 0, 4)):
        pending[source * 16] = value

    def read_u32(address, count):
        values = {
            ("s1", "num_valid"): [3],
            ("s1", "tile_row_source"): [0, 3, 0x00FFFFFF],
            ("s2", "rank_token_pending"): pending,
            ("s2", "node_dest_rank_mask"): [1, 0, 2, 0],
            ("s2", "rank_reduce_queue_tail"): [2],
            ("s2", "rank_reduce_queue"): [2, 0],
            ("s2", "rank_return_count"): [1, 1, 2],
        }[address]
        return values[:count]

    def read_u64(address, count):
        assert address == ("s2", "rank_token_ready")
        return [7, 7, 6, 6][:count]

    monkeypatch.setattr(cco, "read_window_u32", read_u32)
    monkeypatch.setattr(cco, "read_window_u64", read_u64)
    operator = SimpleNamespace(
        stage2_layout=SimpleNamespace(include_rank_partials=True),
        world_size=2,
        mtpr=2,
        stage2_rail_return_schedule="compact",
    )

    state = MegaMoETileA4W4CommProbe._debug_rank_local_completion(
        operator,
        generation=7,
        s1_ptr=lambda name: ("s1", name),
        s2_ptr=lambda name: ("s2", name),
    )

    assert state == {
        "rank_local_active_tokens": 2,
        "rank_local_pending_nonzero": 1,
        "rank_local_pending_nonzero_all": 2,
        "rank_local_ready_missing": 1,
        "rank_local_ready_unexpected": 1,
        "rank_reduce_queue_expected": 2,
        "rank_reduce_queue_count": 2,
        "rank_reduce_queue_tail": 2,
        "rank_reduce_queue_head": 0,
        "rank_reduce_queue_permutation_mismatch": 0,
    }


def test_rank_local_stress_flags_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stress_megamoe_tile_ep16_sparse_routes.py",
            "--stage2-expected-sweep",
            "--route-store-stage2",
            "--rank-local-stage2",
        ],
    )

    with pytest.raises(SystemExit) as error:
        stress_main()
    assert error.value.code == 2


def test_rank_local_stress_requires_stage2_sweep(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stress_megamoe_tile_ep16_sparse_routes.py",
            "--rank-local-stage2",
        ],
    )

    with pytest.raises(SystemExit) as error:
        stress_main()
    assert error.value.code == 2


def test_rank_local_stress_wires_grid_final_and_poison_checks():
    source = " ".join(inspect.getsource(stress_main).split())

    assert 'deferred_stage2.add_argument("--rank-local-stage2"' in source
    assert '"rank_local" if args.rank_local_stage2' in source
    assert "176 if args.route_store_stage2 or args.rank_local_stage2" in source
    assert "128 if args.rank_local_stage2 else 128 * 28" in source
    for field in (
        "rank_local_active_tokens",
        "rank_local_pending_nonzero",
        "rank_local_pending_nonzero_all",
        "rank_local_ready_missing",
        "rank_local_ready_unexpected",
        "rank_reduce_queue_head",
    ):
        assert field in source

    poison_source = inspect.getsource(
        MegaMoETileA4W4CommProbe.poison_stage2_buffers
    )
    for region in (
        "rank_accumulator",
        "rank_token_pending",
        "rank_token_ready",
        "rank_reduce_queue_head",
    ):
        assert region in poison_source
    assert "old_generation" in poison_source

    runner = (
        REPO_ROOT
        / "scripts/megamoe_tile/run_stage2_route_store_stress_ep16.sh"
    ).read_text()
    assert "rank) stage2_mode_flag=--rank-local-stage2" in runner
    assert '"${stage2_mode_flag}"' in runner
    assert 'node_reduce_work_schedule="${7:-static_strided}"' in runner
    assert 'node_reduce_rejoin_blocks="${8:-0}"' in runner


def test_breakdown_plan_accepts_rank_local(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_megamoe_tile_ep16_stage2_breakdown.py",
            "--path",
            "candidate",
            "--plan-only",
            "--stage2-workers",
            "176",
            "--candidate-node-accumulation-mode",
            "rank_local",
            "--candidate-node-reduce-blocks",
            "16",
            "--candidate-node-reduce-vec-bytes",
            "8",
            "--candidate-node-reduce-work-schedule",
            "dynamic_head",
            "--candidate-node-reduce-rejoin-blocks",
            "8",
            "--candidate-rail-return-schedule",
            "compact",
        ],
    )

    assert breakdown_main() == 0
    line = capsys.readouterr().out.strip()
    prefix = "MEGAMOE_EP16_STAGE2_BREAKDOWN_PLAN "
    assert line.startswith(prefix)
    plan = json.loads(line[len(prefix) :])
    assert plan["candidate_node_accumulation_mode"] == "rank_local"
    assert plan["candidate_node_reduce_blocks"] == 16
    assert plan["candidate_node_reduce_vec_bytes"] == 8
    assert plan["candidate_node_reduce_work_schedule"] == "dynamic_head"
    assert plan["candidate_node_reduce_rejoin_blocks"] == 8


def test_breakdown_plan_accepts_rank_local_vec16(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_megamoe_tile_ep16_stage2_breakdown.py",
            "--path",
            "candidate",
            "--plan-only",
            "--candidate-node-accumulation-mode",
            "rank_local",
            "--candidate-node-reduce-vec-bytes",
            "16",
        ],
    )

    assert breakdown_main() == 0
    line = capsys.readouterr().out.strip()
    prefix = "MEGAMOE_EP16_STAGE2_BREAKDOWN_PLAN "
    plan = json.loads(line[len(prefix) :])
    assert plan["candidate_node_reduce_vec_bytes"] == 16


def test_breakdown_rejects_vec16_outside_rank_local(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_megamoe_tile_ep16_stage2_breakdown.py",
            "--path",
            "candidate",
            "--plan-only",
            "--candidate-node-accumulation-mode",
            "route_store",
            "--candidate-node-reduce-vec-bytes",
            "16",
        ],
    )

    with pytest.raises(ValueError, match="requires rank_local"):
        breakdown_main()


def test_validation_and_stress_reject_vec16_outside_rank_local(monkeypatch):
    validation_source = inspect.getsource(validation_main)
    stress_source = inspect.getsource(stress_main)
    assert 'choices=(4, 8, 16), default=8' in validation_source
    assert "choices=(4, 8, 16)" in stress_source
    assert "default=8" in stress_source

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_megamoe_tile_route_store_ep16.py",
            "--mode",
            "route",
            "--output-dir",
            "/tmp/unused",
            "--node-reduce-vec-bytes",
            "16",
        ],
    )
    with pytest.raises(SystemExit) as validation_error:
        validation_main()
    assert validation_error.value.code == 2

    monkeypatch.setattr(
        "sys.argv",
        [
            "stress_megamoe_tile_ep16_sparse_routes.py",
            "--stage2-expected-sweep",
            "--route-store-stage2",
            "--stage2-node-reduce-vec-bytes",
            "16",
        ],
    )
    with pytest.raises(SystemExit) as stress_error:
        stress_main()
    assert stress_error.value.code == 2


def test_breakdown_rejects_rejoin_larger_than_gmm_cta_pool(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_megamoe_tile_ep16_stage2_breakdown.py",
            "--path",
            "candidate",
            "--plan-only",
            "--stage2-workers",
            "39",
            "--candidate-final-combine-blocks",
            "14",
            "--candidate-node-accumulation-mode",
            "rank_local",
            "--candidate-node-reduce-blocks",
            "16",
            "--candidate-node-reduce-work-schedule",
            "dynamic_head",
            "--candidate-node-reduce-rejoin-blocks",
            "16",
            "--candidate-rail-return-schedule",
            "compact",
        ],
    )

    with pytest.raises(ValueError, match="available GMM2 CTA count"):
        breakdown_main()


def test_breakdown_rejects_rejoin_without_queue_publishers(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_megamoe_tile_ep16_stage2_breakdown.py",
            "--path",
            "candidate",
            "--plan-only",
            "--candidate-mode",
            "gmm2_only",
            "--stage2-workers",
            "176",
            "--candidate-node-accumulation-mode",
            "rank_local",
            "--candidate-node-reduce-blocks",
            "16",
            "--candidate-node-reduce-work-schedule",
            "dynamic_head",
            "--candidate-node-reduce-rejoin-blocks",
            "8",
            "--candidate-rail-return-schedule",
            "compact",
        ],
    )

    with pytest.raises(ValueError, match="queue-producing"):
        breakdown_main()


def test_breakdown_shell_appends_rejoin_positionals():
    source = (
        REPO_ROOT / "scripts/megamoe_tile/run_stage2_breakdown_ep16.sh"
    ).read_text()

    assert 'candidate_node_reduce_work_schedule="${29:-static_strided}"' in source
    assert 'candidate_node_reduce_rejoin_blocks="${30:-0}"' in source


def test_validation_shell_appends_rank_local_scheduler_positionals():
    source = (
        REPO_ROOT
        / "scripts/megamoe_tile/run_stage2_route_store_validation_ep16.sh"
    ).read_text()

    assert 'node_reduce_load_schedule="${8:-interleaved}"' in source
    assert 'node_reduce_work_schedule="${9:-static_strided}"' in source
    assert 'node_reduce_rejoin_blocks="${10:-0}"' in source
    assert 'rank_accumulation_mode="${12:-atomic}"' in source
    assert '--node-reduce-work-schedule "${node_reduce_work_schedule}"' in source
    assert '--node-reduce-rejoin-blocks "${node_reduce_rejoin_blocks}"' in source
    assert '--rank-accumulation-mode "${rank_accumulation_mode}"' in source

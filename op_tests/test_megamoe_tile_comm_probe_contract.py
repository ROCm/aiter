# SPDX-License-Identifier: MIT
"""Static safety contract for the diagnostic EP16 communication probe."""

from __future__ import annotations

import importlib
import inspect

import torch

from op_tests.multigpu_tests import megamoe_tile_comm_probe_factory as probe_module
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_comm_only import (
    FusedCommunicationProbePath,
    SplitMoriCommunicationPath,
    _lightweight_shared_inputs,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    HipStageTimer,
    _prepare_local_a4w4,
    _run_local_h1,
    _run_local_h2,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage1 import (
    DIAGNOSTIC_CONTROL_GENERATION_BITS,
    DIAGNOSTIC_PHASE_IDS,
    compile_megamoe_tile_ep16_stage1,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage2 import (
    compile_megamoe_tile_ep16_stage2_a4w4,
)
from aiter.ops.flydsl.kernels.megamoe_tile.kernels.quant_core import build_megamoe_tile_quant_core
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    MegaMoETileA4W4CommProbe,
    MegaMoETileA4W4SparseRealGmm2,
    STAGE1_FULL_GMM1_SUFFIX,
    STAGE1_PROBE_SUFFIX,
    STAGE2_PROBE_SUFFIX,
    cco_flush_batch_contract,
    cco_mori64x2_contract,
    quant_two_cta_assignment,
    split64x2_tilepipe_assignment,
    split128x2_assignment,
    tilepipe_assignment,
    wave64_assignment,
)


def test_megamoe_tile_kernel_imports_use_only_canonical_flydsl_modules():
    canonical_stage1 = importlib.import_module(
        "aiter.ops.flydsl.kernels.megamoe_tile.stage1"
    )
    canonical_stage2 = importlib.import_module(
        "aiter.ops.flydsl.kernels.megamoe_tile.stage2"
    )
    assert "aiter/ops/flydsl/kernels/megamoe_tile/stage1.py" in (
        inspect.getsourcefile(canonical_stage1.compile_megamoe_tile_ep16_stage1)
        or ""
    )
    canonical_public = importlib.import_module(
        "aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4"
    )
    canonical_s1_abi = importlib.import_module(
        "aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi"
    )
    canonical_s2_abi = importlib.import_module(
        "aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi"
    )
    assert canonical_public._STAGE1_MODULE == canonical_stage1.__name__
    assert canonical_public._STAGE2_MODULE == canonical_stage2.__name__
    assert Stage1ArenaLayout is canonical_s1_abi.Stage1ArenaLayout
    assert canonical_s2_abi.Stage2ArenaLayout.__module__ == canonical_s2_abi.__name__


def test_production_stage1_defaults_to_full_gmm_path():
    parameters = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters
    assert parameters["diagnostic_comm_only"].default is False
    assert parameters["tile_pipeline"].default is False
    assert parameters["tile_pipeline_instrument"].default is False
    assert parameters["tile_pipeline_fanout_shards"].default == 16
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert "if const_expr(not diagnostic_comm_only):" in source
    assert '"_diagnostic_comm_only" if diagnostic_comm_only else ""' in source


def test_blockidx_quant_core_has_no_ticket_metadata_or_ready_protocol():
    source = inspect.getsource(build_megamoe_tile_quant_core)
    assert "token = fx.block_idx.x" in source
    assert "block=(BLOCK, 1, 1)" in source
    assert "grid=(fx.Int64(m), 1, 1)" in source
    assert "entry_count" not in source
    assert "topk" not in source
    assert "wait_ready" not in source
    assert "store_i64_global_system" not in source


def test_probe_requests_true_comm_boundary_and_independent_symbols():
    source = inspect.getsource(
        MegaMoETileA4W4CommProbe._build_stage1_probe_launcher
    )
    assert "diagnostic_comm_only=not rejoin" in source
    assert "STAGE1_PROBE_SUFFIX" in source
    assert "_unique_kernel_symbol(suffix)" in source
    source = inspect.getsource(
        MegaMoETileA4W4CommProbe._build_stage2_probe_launcher
    )
    assert 'suffix = f"{STAGE2_PROBE_SUFFIX}_{self.stage2_mode}"' in source
    assert "_unique_kernel_symbol(suffix)" in source
    assert STAGE1_PROBE_SUFFIX != STAGE2_PROBE_SUFFIX


def test_sparse_real_gmm2_factory_cannot_select_zero_compute():
    init_source = inspect.getsource(MegaMoETileA4W4SparseRealGmm2.__init__)
    stage2_source = inspect.getsource(
        MegaMoETileA4W4SparseRealGmm2._build_stage2_probe_launcher
    )
    assert '"stage1_mode": "internodev1_tilepipe"' in init_source
    assert '"stage1_cco_geometry": "sparse_wqe"' in init_source
    assert "MegaMoETileA4W4._compile_stage2(self)" in stage2_source
    assert "launcher.gemm2_contraction = True" in stage2_source
    assert '"zero_gemm2" in launcher.kernel_name' in stage2_source


def test_mori_reference_can_expand_all_local_topk_routes():
    prepare_source = inspect.getsource(_prepare_local_a4w4)
    h1_source = inspect.getsource(_run_local_h1)
    h2_source = inspect.getsource(_run_local_h2)
    assert "expand_local_routes" in prepare_source
    assert "route_rows, route_slots = torch.nonzero" in prepare_source
    assert "(source_slots << 24) | source_rows" in prepare_source
    assert 'topk=context["gmm1_topk"]' in h1_source
    assert 'topk=context["gmm2_topk"]' in h2_source
    assert 'epilog=context["epilog"]' in h2_source


def test_stage2_probe_keeps_direct_epilogue_work_shape():
    source = inspect.getsource(probe_module)
    assert "range(BM // 16)" in source
    assert "range((BN // 4) // 16)" in source
    class_source = inspect.getsource(
        MegaMoETileA4W4CommProbe._build_stage2_probe_launcher
    )
    assert "preserves_direct_lsa_atomic_epilogue = True" in class_source


def test_probe_snapshot_is_protocol_only():
    source = inspect.getsource(MegaMoETileA4W4CommProbe.debug_direct_tile_snapshot)
    assert "read_window_bytes" not in source
    assert "canonical" not in source
    assert '"snapshot_kind": "protocol_counters_only"' in source


def test_probe_precompile_is_serial_and_never_launches_gpu_work():
    module_source = inspect.getsource(probe_module)
    assert "jit_function_impl._build_call_state" in module_source
    assert "launcher._call_state_cache.clear()" in module_source
    source = inspect.getsource(MegaMoETileA4W4CommProbe._compile_stage1)
    assert "for turn in range(self.gpus_per_node):" in source
    assert "if self.local_rank == turn:" in source
    assert source.count("dist.barrier()") == 2
    assert "_precompile_without_launch(stage1" in source
    assert "_precompile_without_launch(stage2" in source


def test_probe_only_inputs_skip_standalone_quant_jit():
    source = inspect.getsource(_lightweight_shared_inputs)
    assert "if quantize_for_mori:" in source
    assert "per_1x32_f4_quant" in source
    assert "The fused probe consumes x_bf16" in source


def test_split_mori_local_prepare_reuses_gemm_output_workspace():
    source = inspect.getsource(SplitMoriCommunicationPath._prepare_stage)
    assert "output_workspace=self._prepare_workspace" in source
    assert 'for name in ("inter_q", "inter_s", "hidden_dummy")' in source


def test_stage_timer_materializes_events_before_begin_iteration():
    init_source = inspect.getsource(HipStageTimer.__init__)
    assert "self._prime_events()" in init_source
    prime_source = inspect.getsource(HipStageTimer._prime_events)
    assert "event.record(stream)" in prime_source
    assert "stream.synchronize()" in prime_source


def test_probe_prime_has_stage_synchronization_boundaries():
    source = inspect.getsource(FusedCommunicationProbePath.prime_and_check)
    assert '"phase=stage1_launch"' in source
    assert "debug_stage1_comm_snapshot" in source
    assert '"phase": "stage1_done"' in source
    assert '"phase=stage2_launch"' in source
    assert '"phase=stage2_done"' in source
    # Four protocol boundaries plus optional out-of-band quant-core and
    # prequant/internal-record canonical checks.
    assert source.count("torch.cuda.synchronize") == 6
    assert source.count("dist.barrier()") == 5


def test_prequant_stage1_keeps_legacy_default_and_explicit_scale_pointer():
    parameters = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters
    assert parameters["prequant_input"].default is False
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert '"_prequant_input" if prequant_input else ""' in source
    assert (
        '"scoreboard_v14_tilequeue_rankslots_payload2_qpballot4_no_send_atomic_inputscale_abi3"'
        in source
    )
    assert "input_scale: fx.Int64" in source
    assert "input_q_dw" in source
    assert "prequant_input currently requires full Stage1" in source
    base_source = inspect.getsource(
        __import__(
            "aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4",
            fromlist=["MegaMoETileA4W4"],
        ).MegaMoETileA4W4._launch_stage1
    )
    assert "0 if input_scale is None else input_scale.data_ptr()" in base_source
    factory_source = inspect.getsource(MegaMoETileA4W4CommProbe._compile_stage1)
    assert "internal_quant_reference" in factory_source
    init_source = inspect.getsource(FusedCommunicationProbePath.__init__)
    assert "dynamic_per_group_scaled_quant" in init_source
    assert "self._prequant_q = torch.empty" in init_source
    assert "self._prequant_scale = torch.empty" in init_source
    validation_source = inspect.getsource(
        FusedCommunicationProbePath._validate_prequant_buffers
    )
    for contract in (
        "expected_q_shape",
        "expected_scale_shape",
        "is_contiguous",
        "storage_offset",
        "element_size",
        "untyped_storage().nbytes()",
        "data_ptr() % 16",
    ):
        assert contract in validation_source
    launch_source = inspect.getsource(FusedCommunicationProbePath._launch_pair)
    assert "per_1x32_f4_quant_hip" not in launch_source
    assert "self._run_prequant()" in launch_source
    assert 'timer.stage("prequant_bf16_to_a4_hip", prequant)' in launch_source
    assert '"stage1_prequant_hier_dispatch_scoreboard"' in launch_source


def test_stage2_diagnostic_modes_are_compile_time_and_default_full():
    parameter = inspect.signature(
        compile_megamoe_tile_ep16_stage2_a4w4
    ).parameters["diagnostic_mode"]
    assert parameter.default == "full"
    source = inspect.getsource(compile_megamoe_tile_ep16_stage2_a4w4)
    assert 'diagnostic_mode == "atomic_only"' in source
    assert 'diagnostic_mode == "return_only"' in source
    assert "role_enabled" in source
    assert "compute_enabled" in source
    assert "wait_ready(" in source
    assert '"diagnostic_mode": diagnostic_mode' in source


def test_split128x2_ticket_map_covers_each_plane_and_destination_once():
    for domain in ("remote", "local"):
        for dest in range(8):
            seen = []
            producers = 0
            for ticket in range(256):
                got_domain, got_dest, shard, tokens = split128x2_assignment(ticket)
                if got_domain == domain and got_dest == dest:
                    producers += 1
                    assert 0 <= shard < 16
                    seen.extend(tokens)
            assert producers == 16
            assert sorted(seen) == list(range(128))
            assert len(seen) == len(set(seen))


def test_split128x2_uses_arena_generation_flags_and_rejoin_contract():
    parameter = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters[
        "diagnostic_split_fanout"
    ]
    assert parameter.default is False
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert "internodev1_split128x2_grid256" in source
    assert "split128x2_rejoin256_posteos" in source
    assert 'local_addr("fanout_done")' in source
    assert "is_intra_fanout" in source
    assert "post_eos_static_strided_256_all_roles_rejoin" in source
    factory_source = inspect.getsource(
        MegaMoETileA4W4CommProbe._build_stage1_probe_launcher
    )
    assert 'region("fanout_done")' in factory_source
    layout = Stage1ArenaLayout.create()
    fanout = layout.region("fanout_done")
    assert fanout.shape == (2, 8, 32)
    assert fanout.dtype == __import__("torch").int64
    assert fanout.offset > layout.region("error_count").offset


def test_split64x2_tilepipe_reserves_128_early_compute_ctas():
    active = set()
    for domain in ("remote", "local"):
        for dest in range(8):
            seen = []
            producers = 0
            for ticket in range(256):
                got_domain, got_dest, shard, tokens = (
                    split64x2_tilepipe_assignment(ticket)
                )
                if got_domain == domain and got_dest == dest:
                    active.add(ticket)
                    producers += 1
                    assert 0 <= shard < 8
                    seen.extend(tokens)
            assert producers == 8
            assert sorted(seen) == list(range(128))
            assert len(seen) == len(set(seen))
    assert len(active) == 128
    assert all(
        split64x2_tilepipe_assignment(ticket)[0] is None
        for ticket in (*range(64, 128), *range(192, 256))
    )


def test_tilepipe_tunable_fanout_maps_each_token_once():
    for fanout_shards in (8, 12, 16):
        active = set()
        for domain in ("remote", "local"):
            for dest in range(8):
                seen = []
                for ticket in range(256):
                    got_domain, got_dest, shard, tokens = tilepipe_assignment(
                        ticket, fanout_shards
                    )
                    if got_domain == domain and got_dest == dest:
                        active.add(ticket)
                        assert 0 <= shard < fanout_shards
                        seen.extend(tokens)
                assert sorted(seen) == list(range(128))
                assert len(seen) == len(set(seen))
        assert len(active) == 16 * fanout_shards


def test_sparse_tile_pipeline_publishes_full_tiles_before_eos():
    parameters = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters
    assert parameters["tile_pipeline"].default is False
    assert parameters["tile_pipeline_instrument"].default is False
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert "tile_pipeline requires the full real-GMM1" in source
    assert 'completed == fx.Int32(BM - 1)' in source
    assert '_enqueue_tile_jobs(dest, physical, True)' in source
    assert 'local_addr("h1_ready_queue_generation")' in source
    assert 'local_addr("h1_queue_eos")' in source
    assert '"early_full_tile_enqueue": bool(tile_pipeline)' in source
    assert '"concurrent_ready_queue_8_shards_256_all_roles_rejoin"' in source
    assert STAGE1_FULL_GMM1_SUFFIX != STAGE1_PROBE_SUFFIX
    validator_source = inspect.getsource(
        __import__(
            "op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel",
            fromlist=["_validate_direct_tile_debug_snapshot"],
        )._validate_direct_tile_debug_snapshot
    )
    assert 'snapshot.get("tile_pipeline", False)' in validator_source
    assert 'snapshot.get("queue_permutation_mismatch", -1)' in validator_source


def test_wave64_ticket_map_covers_each_plane_and_destination_once():
    active_waves = 0
    for domain in ("remote", "local"):
        for dest in range(8):
            seen = []
            ctas = set()
            producers = 0
            for ticket in range(256):
                for wave in range(4):
                    got_domain, got_dest, shard, tokens = wave64_assignment(
                        ticket, wave
                    )
                    if got_domain == domain and got_dest == dest:
                        ctas.add(ticket)
                        assert 0 <= shard < 16
                        if tokens:
                            active_waves += 1
                            producers += 1
                            seen.extend(tokens)
            assert producers == 16
            assert len(ctas) == 16
            assert sorted(seen) == list(range(128))
            assert len(seen) == len(set(seen))
    assert active_waves == 2 * 8 * 16


def test_wave64_is_diagnostic_and_keeps_production_default_off():
    parameters = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters
    assert parameters["diagnostic_wave_fanout"].default is False
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert "diagnostic_wave_fanout requires split fanout mode" in source
    assert "def _dispatch_record_wave" in source
    assert "rocdl.readlane" in source
    assert (
        "128_inter_plus_128_intra_ctas_one_active_wave_eight_records"
        in source
    )


def test_cco_flush_batches_cover_every_chunk_qp_and_wait_request_once():
    expected_coverage = {
        (chunk, qp) for chunk in range(8) for qp in range(4)
    }
    for batch_size in (1, 2, 4, 8):
        contract = cco_flush_batch_contract(batch_size)
        assert set(contract["data_ready_coverage"]) == expected_coverage
        assert set(contract["credit_coverage"]) == expected_coverage
        assert len(contract["data_ready_coverage"]) == 32
        assert len(contract["credit_coverage"]) == 32
        assert len(set(contract["request_slots"])) == 4 * (8 // batch_size)
        assert contract["logical_doorbells"] == 2 * 4 * (8 // batch_size)
        assert contract["data_wqes"] == 64
        assert contract["credit_wqes"] == 32


def test_production_cco_flush_default_is_one_and_symbol_is_conditional():
    parameter = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters[
        "cco_chunks_per_flush"
    ]
    assert parameter.default == 1
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert 'cco_chunks_per_flush not in (1, 2, 4, 8)' in source
    assert 'f"_cco_flushb{cco_chunks_per_flush}"' in source
    assert "request_index" in source


def test_stage1_phase_modes_are_compile_time_and_setup_precedes_stage_event():
    phase = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters[
        "diagnostic_phase"
    ]
    assert phase.default == "full"
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    for name in (
        "quant_core_only",
        "quant_pack_only",
        "transport_only",
        "fanout_only",
        "dispatch_only",
    ):
        assert name in source
    assert "non-full diagnostic_phase requires split comm-only mode" in source
    assert 'diagnostic_phase == "transport_only"' in source
    assert 'local_addr("remote_chunk_consumed")' in source
    assert "control_generation" in source
    assert "DIAGNOSTIC_CONTROL_GENERATION_BITS" in source

    factory_source = inspect.getsource(
        MegaMoETileA4W4CommProbe._compile_stage1
    )
    assert '_build_stage1_probe_launcher("quant_pack_only")' in factory_source
    assert '_build_stage1_probe_launcher("transport_only")' in factory_source

    run_source = inspect.getsource(FusedCommunicationProbePath._launch_pair)
    assert run_source.index("phase_setup()") < run_source.index(
        "timer.stage(self.stage_names[0], stage1)"
    )


def test_legacy_dispatch_record_finishes_with_converged_cta_barrier():
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    marker = "        def _dispatch_record_wave(record, dest, source_index):"
    before_wave_helper = source[: source.index(marker)]
    tail = before_wave_helper.rsplit("_ = completed", 1)[1]
    assert tail.startswith("\n            gpu.barrier()")


def test_stage1_diagnostic_control_generations_do_not_alias():
    phase_ids = [
        value
        for name, value in DIAGNOSTIC_PHASE_IDS.items()
        if name != "full"
    ]
    stride = 1 << DIAGNOSTIC_CONTROL_GENERATION_BITS
    assert len(set(phase_ids)) == len(phase_ids)
    assert min(phase_ids) >= 0
    assert max(phase_ids) < stride
    for generation in range(1, 4):
        current = [generation * stride + phase for phase in phase_ids]
        following = [
            (generation + 1) * stride + phase for phase in phase_ids
        ]
        assert len(set(current)) == len(current)
        assert max(current) < min(following)


def test_phase_payload_and_protocol_geometry():
    layout = Stage1ArenaLayout.create()
    assert layout.max_tokens * layout.wire.record_bytes == 512 * 1024
    assert layout.dispatch_chunks == 8
    assert layout.num_qp == 4


def test_mori64x2_geometry_covers_two_contiguous_halves():
    contract = cco_mori64x2_contract()
    assert contract["payload_bytes_per_half"] == 262144
    assert contract["payload_bytes_per_rank"] == 524288
    assert contract["ready_indices"] == (0, 1)
    assert contract["consumed_indices"] == (0, 1)
    assert contract["credit_indices"] == (0, 1)
    assert contract["request_indices"] == (0, 1)
    assert contract["logical_doorbells"] == 4
    covered = []
    for half in contract["halves"]:
        assert half["qp"] == half["half"]
        covered.extend(range(half["token_begin"], half["token_end"]))
    assert covered == list(range(128))


def test_mori64x2_is_diagnostic_and_production_defaults_chunked():
    parameters = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters
    assert parameters["cco_geometry"].default == "chunked"
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert "mori64x2 geometry requires split fanout" in source
    assert "fx.Int64(64 * record_bytes)" in source
    assert '"cco_geometry": cco_geometry' in source


def test_sparse_wqe_transport_has_per_token_and_per_qp_completion_state():
    layout = Stage1ArenaLayout.create()
    # Producers write local per-token 0/1 decisions; one coordinator wave
    # ballots them into the four terminal QP words.
    assert layout.region("sparse_remote_token_ready").shape == (2, 128)
    assert layout.region("sparse_remote_qp_ready").shape == (2, 4)
    assert layout.region("sparse_remote_request").shape == (2, 4)
    assert layout.region("sparse_remote_batch_ready").shape == (2,)
    assert layout.region("sparse_remote_credit").shape == (2,)
    assert layout.region("sparse_remote_consumed").shape == (2,)
    assert layout.region("sparse_remote_send_count").shape == (2,)
    assert layout.region("h1_early_full_tiles").shape == (2,)
    assert layout.region("h1_gmm_started_before_all_comm_eos").shape == (2,)
    assert layout.region("h1_gmm_completed_before_all_comm_eos").shape == (2,)

    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert 'cco_geometry == "sparse_wqe"' in source
    assert 'local_addr("dispatch_staging_ready")' in source
    assert 'local_addr("sparse_remote_token_ready")' in source
    assert 'window_off("sparse_remote_qp_ready")' in source
    assert "rocdl.ballot" in source
    assert 'local_addr("sparse_remote_batch_ready")' in source
    assert 'local_addr("sparse_remote_consumed")' in source
    assert 'window_off("sparse_remote_credit")' in source
    assert "aggregate=True" in source


def test_sparse_wqe_streams_each_qp_before_the_batch_gate():
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    scan = source.index("for stream_qp in range_constexpr(layout.num_qp)")
    terminal = source.index('window_off("sparse_remote_qp_ready")', scan)
    flush = source.index("request = flush_async(", terminal)
    next_phase = source.index(
        "for ready_qp in range_constexpr(layout.num_qp)", flush
    )
    assert scan < terminal < flush < next_phase
    assert "ready_qp = shard % fx.Int32(layout.num_qp)" in source
    assert "sparse_qp_token_mask" in source
    assert (
        'local_addr("sparse_remote_batch_ready"),\n'
        "                                generation,"
    ) in source
    assert (
        '"four_streamed_qp_terminal_words_with_inter_compute_batch_gate"'
        in source
    )


def test_stage1_record_preserves_duplicate_rank_slots_and_deduplicates_payload():
    layout = Stage1ArenaLayout.create()
    wire = layout.wire
    assert wire.ids_offset == 3808
    assert wire.weights_offset == 3872
    assert wire.source_offset == 3936
    assert wire.route_mask_offset == 3944
    assert wire.rank_slot_masks_offset == 3952
    assert wire.rank_slot_masks_bytes == 32
    assert wire.raw_bytes == 3984
    assert wire.record_bytes == 4096
    assert layout.route_capacity == 16 * 128 * 16
    assert layout.max_route_tiles == 1080
    assert layout.max_route_rows == 34560
    assert layout.max_tiles_per_expert == 1024
    assert layout.region("grouped_input_q").shape == (2, 2048, 3584)

    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert "wire.rank_slot_masks_offset" in source
    assert "route_slots = _record_dest_slot_mask(record, dest)" in source
    assert "fx.Int64(source_index) * fx.Int64(wire.payload_bytes)" in source
    assert "_dispatch_route(" in source


def test_paired_rank_fixture_has_two_routes_per_rank_and_half_remote_tokens():
    shape = BenchmarkShape()
    for rank in (0, 7, 8, 15):
        shared = _lightweight_shared_inputs(
            shape,
            rank,
            16,
            torch.device("cpu"),
            quantize_for_mori=False,
            route_pattern="paired-rank-half-remote",
        )
        ids = shared.topk_ids.cpu()
        weights = shared.route_weights.cpu()
        owners = torch.div(ids, shape.local_experts, rounding_mode="floor")
        assert torch.allclose(weights.sum(dim=1), torch.ones(shape.tokens))
        assert torch.all(weights[:, 0] != weights[:, 1])
        for token in range(shape.tokens):
            counts = torch.bincount(owners[token], minlength=16)
            assert sorted(value for value in counts.tolist() if value) == [2] * 8
            assert torch.unique(ids[token]).numel() == shape.topk
        source_node = rank // shape.gpus_per_node
        remote_tokens = sum(
            int(
                torch.any(
                    owners[token] // shape.gpus_per_node != source_node
                ).item()
            )
            for token in range(shape.tokens)
        )
        assert remote_tokens == 64


def test_paired_rank_local_fixture_keeps_full_route_work_without_remote_payload():
    shape = BenchmarkShape()
    for rank in (0, 8):
        shared = _lightweight_shared_inputs(
            shape,
            rank,
            16,
            torch.device("cpu"),
            quantize_for_mori=False,
            route_pattern="paired-rank-local-only",
        )
        ids = shared.topk_ids.cpu()
        owners = torch.div(ids, shape.local_experts, rounding_mode="floor")
        source_node = rank // shape.gpus_per_node
        assert torch.all(owners // shape.gpus_per_node == source_node)
        for token in range(shape.tokens):
            counts = torch.bincount(owners[token], minlength=16)
            assert sorted(value for value in counts.tolist() if value) == [2] * 8
            assert torch.unique(ids[token]).numel() == shape.topk


def test_paired_rank_remote_prefix_covers_qp_bitmap_boundaries():
    shape = BenchmarkShape()
    for remote_tokens in (0, 1, 2, 3, 4, 31, 32, 33, 63, 64, 127, 128):
        shared = _lightweight_shared_inputs(
            shape,
            0,
            16,
            torch.device("cpu"),
            quantize_for_mori=False,
            route_pattern="paired-rank-remote-prefix",
            remote_token_count=remote_tokens,
        )
        ids = shared.topk_ids.cpu()
        owners = torch.div(ids, shape.local_experts, rounding_mode="floor")
        actual_masks = [0] * 4
        for token in range(shape.tokens):
            counts = torch.bincount(owners[token], minlength=16)
            assert sorted(value for value in counts.tolist() if value) == [2] * 8
            assert torch.unique(ids[token]).numel() == shape.topk
            if torch.any(owners[token] // shape.gpus_per_node == 1):
                actual_masks[token % 4] |= 1 << (token // 4)
        expected_masks = []
        for qp in range(4):
            bit_count = max(0, (remote_tokens + 3 - qp) // 4)
            expected_masks.append((1 << bit_count) - 1)
        assert actual_masks == expected_masks


def test_single_rank_fixture_reaches_arbitrary_topk_capacity_bound():
    shape = BenchmarkShape()
    for hot_rank in (0, 8):
        per_destination = [0] * 16
        for source_rank in range(16):
            shared = _lightweight_shared_inputs(
                shape,
                source_rank,
                16,
                torch.device("cpu"),
                quantize_for_mori=False,
                route_pattern="single-rank-max",
                hot_rank=hot_rank,
            )
            ids = shared.topk_ids.cpu()
            owners = torch.div(ids, shape.local_experts, rounding_mode="floor")
            assert torch.all(owners == hot_rank)
            for token in range(shape.tokens):
                assert torch.unique(ids[token]).numel() == shape.topk
            per_destination[hot_rank] += ids.numel()
        assert per_destination[hot_rank] == 16 * 128 * 16
        assert sum(per_destination) == Stage1ArenaLayout.create().route_capacity


def test_single_expert_fixture_reaches_per_expert_capacity_bound():
    shape = BenchmarkShape()
    total = 0
    for source_rank in range(16):
        shared = _lightweight_shared_inputs(
            shape,
            source_rank,
            16,
            torch.device("cpu"),
            quantize_for_mori=False,
            route_pattern="single-expert-max",
            hot_rank=0,
        )
        assert torch.count_nonzero(shared.topk_ids).item() == 0
        total += shared.topk_ids.numel()
    layout = Stage1ArenaLayout.create()
    assert total == layout.route_capacity
    assert (total + 31) // 32 == layout.max_tiles_per_expert


def test_node_route_count_fixture_covers_every_stage2_expected_value():
    shape = BenchmarkShape()
    for local_routes in range(17):
        shared = _lightweight_shared_inputs(
            shape,
            0,
            16,
            torch.device("cpu"),
            quantize_for_mori=False,
            route_pattern="node-route-count",
            local_route_count=local_routes,
        )
        ids = shared.topk_ids.cpu()
        owners = torch.div(ids, shape.local_experts, rounding_mode="floor")
        assert torch.all(
            (owners // shape.gpus_per_node == 0).sum(dim=1)
            == local_routes
        )
        assert torch.all(
            (owners // shape.gpus_per_node == 1).sum(dim=1)
            == shape.topk - local_routes
        )
        for token in range(shape.tokens):
            assert torch.unique(ids[token]).numel() == shape.topk


def test_stage2_accepts_route_counts_zero_through_topk():
    source = inspect.getsource(compile_megamoe_tile_ep16_stage2_a4w4)
    assert "comm_ops.fence_agent_release()" in source
    assert "expected > fx.Int32(TOPK)" in source
    assert "if expected == fx.Int32(0):" in source
    assert "ready_ptr + fx.Int64(item) * fx.Int64(8), generation" in source
    assert "expected <= fx.Int32(TOPK)" in source


def test_sparse_route_fixture_can_have_zero_remote_tokens():
    source = inspect.getsource(_lightweight_shared_inputs)
    assert 'route_pattern == "local-only-padded"' in source
    assert "torch.full_like(owner, -1)" in source


def test_quant_two_cta_assignment_covers_every_group_once():
    by_token = {token: [] for token in range(128)}
    for ticket in range(256):
        token, half, groups, owns_metadata = quant_two_cta_assignment(ticket)
        assert half in (0, 1)
        assert owns_metadata == (half == 0)
        assert len(groups) == 112
        by_token[token].append((half, groups, owns_metadata))
    for entries in by_token.values():
        assert len(entries) == 2
        assert sorted(entry[0] for entry in entries) == [0, 1]
        all_groups = [g for _, groups, _ in entries for g in groups]
        assert sorted(all_groups) == list(range(224))
        assert sum(int(owner) for _, _, owner in entries) == 1


def test_quant_two_cta_uses_appended_generation_flags_and_defaults_off():
    parameters = inspect.signature(compile_megamoe_tile_ep16_stage1).parameters
    assert parameters["quant_two_cta_per_token"].default is False
    layout = Stage1ArenaLayout.create()
    region = layout.region("quant_half_done")
    assert region.shape == (2, 128)
    assert region.offset > layout.region("fanout_done").offset
    source = inspect.getsource(compile_megamoe_tile_ep16_stage1)
    assert 'local_addr("quant_half_done")' in source
    assert '"quant_two_cta_per_token": bool(' in source

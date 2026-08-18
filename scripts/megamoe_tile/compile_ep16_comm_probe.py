#!/usr/bin/env python3
"""Lower/link the independent EP16 communication-probe code objects."""

from __future__ import annotations

import argparse

import flydsl.expr as fx

from aiter.ops.flydsl.kernels.megamoe_tile import stage1 as stage1_impl
from aiter.ops.flydsl.kernels.megamoe_tile import stage2 as stage2_impl
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import Stage2ArenaLayout
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    STAGE1_FULL_GMM1_SUFFIX,
    STAGE1_PROBE_SUFFIX,
    STAGE2_PROBE_SUFFIX,
    _noop_issue_a_load_lds_dt,
    _precompile_without_launch,
    _unique_kernel_symbol,
    _zero_gemm2_compute_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--worker-blocks", type=int, default=160)
    parser.add_argument(
        "--stage1-mode",
        choices=(
            "legacy",
            "internodev1_split128x2",
            "internodev1_split128x2_no_arrival_rmw",
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64",
            "internodev1_wave64_rejoin",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--stage2-mode",
        choices=("full", "atomic_only", "return_only"),
        default="full",
    )
    parser.add_argument(
        "--cco-chunks-per-flush",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
    )
    parser.add_argument(
        "--stage1-phase",
        choices=(
            "full",
            "quant_core_only",
            "quant_pack_only",
            "transport_only",
            "fanout_only",
            "dispatch_only",
        ),
        default="full",
    )
    parser.add_argument(
        "--cco-geometry",
        choices=("chunked", "mori64x2", "sparse_wqe"),
        default="chunked",
    )
    parser.add_argument("--quant-two-cta-per-token", action="store_true")
    parser.add_argument("--prequant-input", action="store_true")
    parser.add_argument("--tile-pipeline-instrument", action="store_true")
    parser.add_argument(
        "--tile-pipeline-fanout-shards",
        type=int,
        choices=(8, 12, 16),
        default=16,
    )
    args = parser.parse_args()

    s1 = Stage1ArenaLayout.create()
    s2 = Stage2ArenaLayout.create()
    combo = TwoKernelArenaLayout.compose(s1, s2)

    wave_fanout = args.stage1_mode in (
        "internodev1_wave64",
        "internodev1_wave64_rejoin",
    )
    split = (
        args.stage1_mode.startswith("internodev1_split")
        or args.stage1_mode == "internodev1_tilepipe"
        or wave_fanout
    )
    rejoin = args.stage1_mode in (
        "internodev1_split128x2_rejoin",
        "internodev1_tilepipe",
        "internodev1_wave64_rejoin",
    )
    tile_pipeline = args.stage1_mode == "internodev1_tilepipe"
    no_arrival_rmw = (
        args.stage1_mode == "internodev1_split128x2_no_arrival_rmw"
    )
    if tile_pipeline:
        fanout_ctas = 8 * args.tile_pipeline_fanout_shards
        stage1_suffix = (
            f"{STAGE1_FULL_GMM1_SUFFIX}_split{fanout_ctas}x2_"
            "tilepipe256r"
        )
    elif wave_fanout:
        stage1_suffix = (
            f"{STAGE1_FULL_GMM1_SUFFIX}_wave64x1_fullgrid256_rejoin"
            if rejoin
            else f"{STAGE1_PROBE_SUFFIX}_wave64x1_fullgrid256"
        )
    elif split:
        stage1_suffix = (
            f"{STAGE1_FULL_GMM1_SUFFIX}_split128x2_rejoin256_posteos"
            if rejoin
            else (
                f"{STAGE1_PROBE_SUFFIX}_"
                "internodev1_split128x2_grid256_no_arrival_rmw"
                if no_arrival_rmw
                else f"{STAGE1_PROBE_SUFFIX}_internodev1_split128x2_grid256"
            )
        )
    else:
        stage1_suffix = STAGE1_PROBE_SUFFIX
    with _unique_kernel_symbol(stage1_suffix):
        stage1 = stage1_impl.compile_megamoe_tile_ep16_stage1(
            s1,
            s2,
            rank=args.rank,
            stage2_window_offset=combo.stage2_offset,
            worker_blocks=256 if split else args.worker_blocks,
            enable_cco=True,
            diagnostic_comm_only=not rejoin,
            diagnostic_split_fanout=split,
            diagnostic_wave_fanout=wave_fanout,
            diagnostic_no_arrival_rmw=no_arrival_rmw,
            cco_chunks_per_flush=args.cco_chunks_per_flush,
            cco_geometry=args.cco_geometry,
            diagnostic_phase=args.stage1_phase,
            quant_two_cta_per_token=args.quant_two_cta_per_token,
            prequant_input=args.prequant_input,
            tile_pipeline=tile_pipeline,
            tile_pipeline_instrument=args.tile_pipeline_instrument,
            tile_pipeline_fanout_shards=args.tile_pipeline_fanout_shards,
        )

    stage2_impl.issue_a_load_lds_dt = _noop_issue_a_load_lds_dt
    stage2_impl.gemm2_compute_v2 = _zero_gemm2_compute_v2
    stage2_suffix = f"{STAGE2_PROBE_SUFFIX}_{args.stage2_mode}"
    with _unique_kernel_symbol(stage2_suffix):
        stage2 = stage2_impl.compile_megamoe_tile_ep16_stage2_a4w4(
            combo, rank=args.rank, diagnostic_mode=args.stage2_mode
        )

    s1_args = (
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0),
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int32(128), fx.Int64(1),
        fx.Stream(None),
    )
    s2_args = (
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0),
        fx.Int64(1), fx.Int32(128), fx.Int32(args.worker_blocks), fx.Int64(0),
        fx.Stream(None),
    )
    _precompile_without_launch(stage1, s1_args)
    _precompile_without_launch(stage2, s2_args)
    if stage1._call_state_cache or stage2._call_state_cache:
        raise AssertionError("fake no-launch CallState leaked into the hot cache")
    if not stage1._mem_cache or not stage2._mem_cache:
        raise AssertionError("probe precompile did not retain its CompiledArtifact")

    # launcher.kernel_name is the production metadata string; the decorator
    # suffix below is the actual independent GPU symbol/JIT key.
    print(
        "STAGE1_COMM_PROBE_COMPILE_OK",
        f"{stage1.kernel_name}_{stage1_suffix}",
        "LDS",
        stage1.lds_bytes,
    )
    print(
        "STAGE2_COMM_PROBE_COMPILE_OK",
        f"{stage2.kernel_name}_{stage2_suffix}",
        "LDS",
        stage2.lds_bytes,
    )


if __name__ == "__main__":
    main()

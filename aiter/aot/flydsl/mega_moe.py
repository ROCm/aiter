#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT profile bundles for the DeepSeek-V4-Pro MegaMoE A8W4 path."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import flydsl.expr as fx
import torch

from aiter.aot.flydsl.common import compile_only_env, override_env, run_jobs_parallel
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    build_mega_moe_bundle_plan,
)

DEFAULT_MTPRS = (8192, 16384, 32768)
# DeepSeek-V4-Pro deployment profiles: r0, r32, and r64 redundant experts.
# Keep all three in the default AOT job set so the service never falls back to
# an online compile merely because EPLB changes the physical expert count.
DEFAULT_EXPERTS_PER_RANKS = (48, 52, 56)
WORLD_SIZE = 8
TOPK = 6
MODEL_DIM = 7168
INTER_DIM = 3072
NUM_CU = 256
SWIGLU_LIMIT = 10.0
_DEFAULT_COMBINE_BLOCK_NUM = 128
_DEFAULT_COMBINE_WARP_NUM = 8


def _tile_state_stride(mtpr, experts_per_rank):
    """Match ``FlyDSLDispatchGroupMajorOp`` compact metadata capacity."""
    num_valid_max = WORLD_SIZE * mtpr * TOPK + experts_per_rank * 128
    return (num_valid_max + 31) // 32


def default_jobs(
    mtprs=DEFAULT_MTPRS,
    experts_per_ranks=DEFAULT_EXPERTS_PER_RANKS,
):
    return [
        {
            "kernel_name": (
                f"mega_moe_stage{stage}_bundle_mtpr{mtpr}_epr{experts_per_rank}_rank{rank}"
            ),
            "stage": stage,
            "mtpr": mtpr,
            "experts_per_rank": experts_per_rank,
            "rank": rank,
        }
        for mtpr in mtprs
        for experts_per_rank in experts_per_ranks
        for rank in range(WORLD_SIZE)
        for stage in (1, 2)
    ]


def _tensor(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="cpu")


def _compile_stage1(mtpr, experts_per_rank, rank, plan):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_prepare import (
        preload_mega_moe_prepare,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import (
        compile_mega_moe_stage1_bundle,
    )
    from aiter.ops.flydsl.kernels.mega_moe.quant import _get_launcher

    tile_state_stride = _tile_state_stride(mtpr, experts_per_rank)
    scale_dim = MODEL_DIM // 32
    seen_prepare = set()
    for entry in plan.entries:
        config = entry.config.stage1
        prepare_blocks = max(1, min(config.prepare_cu, (entry.token_bucket + 63) // 64))
        quant_groups = entry.token_bucket * scale_dim
        quant_blocks = min(
            NUM_CU,
            config.prepare_quant_cu,
            (quant_groups + 511) // 512,
        )
        for num_quant_cu in sorted({0, quant_blocks}):
            identity = (
                config.sort_block_m,
                config.num_dispatch_cu,
                prepare_blocks,
                num_quant_cu,
                config.payload_chunk_rows,
                config.payload_tile_ready,
            )
            if identity in seen_prepare:
                continue
            seen_prepare.add(identity)
            preload_mega_moe_prepare(
                fx.Int64(0),
                fx.Int32(1),
                fx.Int64(0),
                fx.Int64(0),
                fx.Int64(0),
                fx.Int64(0),
                fx.Int64(0),
                fx.Int64(0),
                fx.Stream(None),
                rank=rank,
                experts_per_rank=experts_per_rank,
                fuse_npes=WORLD_SIZE,
                fuse_topk=TOPK,
                fuse_mtpr=mtpr,
                sort_block_m=config.sort_block_m,
                num_dispatch_cu=config.num_dispatch_cu,
                num_prepare_cu=prepare_blocks,
                num_quant_cu=num_quant_cu,
                quant_cu_capacity=NUM_CU,
                model_dim=MODEL_DIM,
                payload_chunk_rows=config.payload_chunk_rows,
                payload_tile_ready=config.payload_tile_ready,
                tile_state_stride=tile_state_stride,
                fanout_masks=(),
                runtime_fanout=True,
                dynamic_fanout=True,
            )

    # The production E2E path quantizes inside prepare.  Keep the public
    # standalone quantize() launcher in the same AOT profile as well because
    # the validation/per-stage benchmark invokes it to isolate Stage1 timing.
    quant_rows = 2
    quant_groups = quant_rows * scale_dim
    _get_launcher(MODEL_DIM, "fp8")(
        _tensor((quant_rows, MODEL_DIM), torch.bfloat16),
        _tensor((quant_rows, MODEL_DIM), torch.float8_e4m3fn),
        _tensor((quant_rows, scale_dim), torch.uint8),
        quant_rows,
        (quant_groups + 63) // 64,
        fx.Stream(None),
    )

    launch = compile_mega_moe_stage1_bundle(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        rank=rank,
        experts_per_rank=experts_per_rank,
        fuse_npes=WORLD_SIZE,
        fuse_topk=TOPK,
        fuse_cap=WORLD_SIZE * mtpr,
        fuse_mtpr=mtpr,
        fuse_scale_dim=MODEL_DIM // 32,
        fixed_slot_dispatch=plan.fixed_slot_dispatch,
        num_cu=NUM_CU,
        tile_state_stride=tile_state_stride,
        variants=plan.stage1_variants,
        swiglu_limit=SWIGLU_LIMIT,
    )
    launch(
        _tensor((1, INTER_DIM), torch.float8_e4m3fn),
        _tensor((1, MODEL_DIM), torch.float8_e4m3fn),
        # Keep every non-unit dimension larger than one so FlyDSL records the
        # same contiguous-stride signature as the runtime weight tensor.  An
        # all-ones placeholder makes the first dimension look unit-strided and
        # produces an AOT cache key that the real tensor can never reuse.
        _tensor((2, 2, 2), torch.uint8),
        _tensor((1, MODEL_DIM // 128), torch.int32),
        _tensor((2, 2), torch.uint8),
        _tensor((1,), torch.int32),
        _tensor((1,), torch.int32),
        _tensor((2,), torch.int32),
        _tensor((1,), torch.uint8),
        fx.Int32(1),
        fx.Int64(0),
        fx.Int32(1),
        *([fx.Int64(0)] * 6),
        fx.Int32(0),
        fx.Stream(None),
    )


def _compile_stage2(mtpr, experts_per_rank, rank, plan):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
        preload_mega_moe_stage2,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2_aligned_pair import (
        preload_mega_moe_stage2_aligned_pair,
    )

    for key in plan.stage2_variants:
        stage2 = key.config
        row_bytes = (
            MODEL_DIM + MODEL_DIM // 32
            if key.p2p_quant == "fp8_blockwise_1x32"
            else MODEL_DIM * 2
        )
        common = {
            "model_dim": MODEL_DIM,
            "inter_dim": INTER_DIM,
            "experts": experts_per_rank,
            "topk": TOPK,
            "rank": rank,
            "npes": WORLD_SIZE,
            "max_tok": mtpr,
            "recv_cap": WORLD_SIZE * mtpr,
            "comb_inp_nbytes": mtpr * TOPK * row_bytes,
            "HIDDEN_MAX": MODEL_DIM,
            "INTER_MAX": INTER_DIM,
            "cu_num": NUM_CU,
            "p2p_quant_type": key.p2p_quant,
            "fixed_slot_dispatch": key.fixed_slot_dispatch,
        }
        residual = (
            replace(stage2, skew_cu=stage2.persist_cu)
            if stage2.aligned_pair
            else stage2
        )
        preload_mega_moe_stage2(
            *([fx.Int64(0)] * 15),
            WORLD_SIZE * mtpr * TOPK + experts_per_rank * 128,
            fx.Int32(INTER_DIM),
            fx.Int32(MODEL_DIM),
            fx.Stream(None),
            BM=residual.block_m,
            SBM=key.sbm,
            BN=residual.block_n,
            BK=residual.block_k,
            use_nt=residual.use_nt,
            g2_bhoist=residual.b_hoist,
            g2_ascale_pf=residual.ascale_prefetch,
            g2_spart=residual.spatial_partition,
            persist=residual.persist,
            persist_cu=residual.persist_cu,
            persist_strided=residual.persist_strided,
            skew_cu=residual.skew_cu,
            g2_bf16_lds=residual.bf16_lds,
            runtime_pair_skip=stage2.aligned_pair,
            scatter_vec=stage2.pair_scatter_vec if stage2.aligned_pair else 8,
            **common,
        )
        if not stage2.aligned_pair:
            continue
        preload_mega_moe_stage2_aligned_pair(
            *([fx.Int64(0)] * 14),
            WORLD_SIZE * mtpr * TOPK + experts_per_rank * 128,
            fx.Int32(INTER_DIM),
            fx.Int32(MODEL_DIM),
            fx.Stream(None),
            model_dim=MODEL_DIM,
            inter_dim=INTER_DIM,
            experts=experts_per_rank,
            topk=TOPK,
            rank=rank,
            npes=WORLD_SIZE,
            max_tok=mtpr,
            recv_cap=WORLD_SIZE * mtpr,
            comb_inp_nbytes=mtpr * TOPK * row_bytes,
            pair_mask=0,
            runtime_pair=True,
            BM=stage2.pair_block_m,
            SBM=key.sbm,
            BN=stage2.pair_block_n,
            BK=stage2.block_k,
            INTER_MAX=INTER_DIM,
            use_nt=stage2.use_nt,
            cu_num=stage2.pair_cu,
            g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch,
            pair_work_weight=stage2.pair_work_weight,
            dual_accumulator=True,
            scatter_vec=stage2.pair_scatter_vec,
            m_swizzle=True,
        )

    # Stage2's production bundle includes the terminal fused combine kernels.
    # Compile them here as well; otherwise a clean AOT-only service still falls
    # back to JIT after all GEMM2 variants have loaded successfully.
    from aiter.ops.flydsl.kernels.communication_ops_utils import GeometryTuningTable
    from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_kernel import (
        make_combine_jit,
    )

    tuning_path = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "flydsl"
        / "kernels"
        / "mega_moe_tuning_config"
        / "flydsl_gfx950_mi355x_IntraNode_ep8.json"
    )
    tuning = GeometryTuningTable.from_tuning_file(
        tuning_path,
        dtype="fp8_ocp",
        hidden_dim=MODEL_DIM,
        zero_copy=False,
        topk=TOPK,
        local_expert_num=experts_per_rank,
        combine_dtype="bf16",
    )
    seen_combine = set()
    for entry in plan.entries:
        geometry = tuning.lookup("combine", entry.token_bucket)
        block_num, warp_num = geometry or (
            _DEFAULT_COMBINE_BLOCK_NUM,
            _DEFAULT_COMBINE_WARP_NUM,
        )
        blockwise_fp8 = entry.config.p2p_quant == "fp8_blockwise_1x32"
        identity = (block_num, warp_num, blockwise_fp8)
        if identity in seen_combine:
            continue
        seen_combine.add(identity)
        launch = make_combine_jit(
            rank=rank,
            npes=WORLD_SIZE,
            experts_per_token=TOPK,
            hidden_dim=MODEL_DIM,
            max_tok_per_rank=mtpr,
            block_num=block_num,
            warp_num_per_block=warp_num,
            data_type=torch.bfloat16,
            enable_weights=False,
            enable_std_moe=False,
            zero_copy=False,
            skip_stage1=True,
            fp8_direct_cast=False,
            blockwise_fp8_transport=blockwise_fp8,
            # MegaMoE uses the fixed destination-slot contract and only needs
            # one receive-count slot per peer for combine metadata.
            max_recv=WORLD_SIZE,
        )
        launch(
            *([fx.Int64(0)] * 18),
            fx.Int32(entry.token_bucket),
            fx.Stream(None),
        )


def compile_one_config(**job):
    result = {**job, "compile_time": None}
    started = time.time()
    try:
        plan = build_mega_moe_bundle_plan(
            job["mtpr"],
            experts_per_rank=job["experts_per_rank"],
            model_dim=MODEL_DIM,
            inter_dim=INTER_DIM,
        )
        with compile_only_env(), override_env("FLYDSL_GPU_ARCH", "gfx950"):
            if job["stage"] == 1:
                _compile_stage1(job["mtpr"], job["experts_per_rank"], job["rank"], plan)
            else:
                _compile_stage2(job["mtpr"], job["experts_per_rank"], job["rank"], plan)
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtpr", type=int, nargs="+", default=list(DEFAULT_MTPRS))
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPERTS_PER_RANKS),
        help="deployment profiles to compile (for example 48 52 56 for r0/r32/r64)",
    )
    args = parser.parse_args()
    jobs = default_jobs(tuple(args.mtpr), tuple(args.experts_per_rank))
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failed} ok, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

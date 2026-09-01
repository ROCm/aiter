#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT profile bundles for the DeepSeek-V4-Pro MegaMoE A8W4 path."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import flydsl.compiler as flyc
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
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import (
        compile_mega_moe_stage1_bundle,
    )
    from aiter.ops.flydsl.kernels.mega_moe.quant import _get_launcher

    # ``forward`` quantizes BF16 activations before the fused Stage1 launch.
    # Compile that runtime signature with the same profile instead of leaving
    # an AOT-only service to fall back to JIT on its first request.
    scale_dim = MODEL_DIM // 32
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
        variants=plan.stage1_variants,
        swiglu_limit=SWIGLU_LIMIT,
    )
    # Compile the complete bundle without selecting or launching a runtime
    # variant.  Passing variant 0 here specializes the AOT artifact to the
    # first entry while the cache identity is shared by every entry; loading
    # that artifact for another runtime variant can use incompatible launch
    # geometry and deadlock the cross-rank protocol.
    flyc.compile(
        launch,
        _tensor((1, INTER_DIM), torch.float8_e4m3fn),
        _tensor((1, MODEL_DIM), torch.float8_e4m3fn),
        # Preserve the real contiguous rank-3 stride signature.  Unit-only
        # dimensions collapse strides and produce an AOT key runtime weights
        # cannot reuse.
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
        fx.Int32(-1),
        fx.Stream(None),
    )


def _compile_stage2(mtpr, experts_per_rank, rank, plan):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
        compile_mega_moe_stage2_bundle,
    )

    p2p_quants = {entry.config.p2p_quant for entry in plan.entries}
    if len(p2p_quants) != 1:
        raise ValueError("one MegaMoE AOT bundle cannot mix Stage2 wire formats")
    p2p_quant = p2p_quants.pop()
    row_bytes = (
        MODEL_DIM + MODEL_DIM // 32
        if p2p_quant == "fp8_blockwise_1x32"
        else MODEL_DIM * 2
    )
    launch = compile_mega_moe_stage2_bundle(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=experts_per_rank,
        topk=TOPK,
        rank=rank,
        npes=WORLD_SIZE,
        max_tok=mtpr,
        recv_cap=WORLD_SIZE * mtpr,
        comb_inp_nbytes_by_quant=((p2p_quant, mtpr * TOPK * row_bytes),),
        HIDDEN_MAX=MODEL_DIM,
        INTER_MAX=INTER_DIM,
        cu_num=NUM_CU,
        variants=plan.stage2_variants,
    )
    # As in Stage1, -1 is the compile-only sentinel for the bundle.  No
    # communication kernel is launched and no runtime variant is specialized.
    flyc.compile(
        launch,
        *([fx.Int64(0)] * 11),
        fx.Int32(1),
        fx.Int32(1),
        fx.Int32(INTER_DIM),
        fx.Int32(MODEL_DIM),
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(-1),
        fx.Stream(None),
    )

    # Stage2 terminates in the fused no-Stage1 combine kernel.  It is outside
    # the GEMM2 bundle, so include every distinct production launch geometry in
    # this stage's AOT profile.
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
        identity = block_num, warp_num, blockwise_fp8
        if identity in seen_combine:
            continue
        seen_combine.add(identity)
        combine = make_combine_jit(
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
            max_recv=WORLD_SIZE,
        )
        combine(
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

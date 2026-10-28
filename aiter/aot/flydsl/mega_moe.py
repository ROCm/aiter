#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT profile bundles for MegaMoE A8W4 and A4W4 deployment shapes, plus the opt-in small-/large-input kernels."""

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
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm import LARGE_M_MTPR

DEFAULT_MTPRS = (8192, 16384, 32768)
DEFAULT_QUANTS = ("a8w4", "a4w4")
# DeepSeek-V4-Pro deployment profiles: r0, r32, and r64 redundant experts.
# Keep all three in the default AOT job set so the service never falls back to
# an online compile merely because EPLB changes the physical expert count.
DEFAULT_EXPERTS_PER_RANKS = (48, 52, 56)
# Kimi-K3 deployment profile (EP8 / epr112, topk16, d3584 / i3072).  Added to
# the default AOT job set so K3 serving never falls back to an online compile.
K3_EXPERTS_PER_RANKS = (112,)
K3_TOPK = 16
K3_MODEL_DIM = 3584
K3_INTER_DIM = 3072
# K3 uses SiTUv2 (hidden_act="situ"); tanh self-saturates so no hard clamp.
# beta=4.0 / linear_beta=25.0 are the K3 checkpoint constants the MegaMoE SiTU
# kernel bakes in (SGLang kimi_k3.py asserts exactly these); they MUST match
# op_tests kimi_k3_route or the AOT bundle name will not match at serving.
K3_ACT = "situv2"
K3_SITU_BETA = 4.0
K3_SITU_LINEAR_BETA = 25.0
WORLD_SIZE = 8
TOPK = 6
MODEL_DIM = 7168
INTER_DIM = 3072
NUM_CU = 256
SWIGLU_LIMIT = 10.0
# Opt-in small-/large-input kernels (AITER_MEGA_SMALL_M / AITER_MEGA_LARGE_M): DeepSeek-V4-Pro EP8 a8w4 only.
SMALL_M_MTPRS = (256, 512)
FUSED_PATH_EXPERTS_PER_RANK = 48
_DEFAULT_COMBINE_BLOCK_NUM = 128
_DEFAULT_COMBINE_WARP_NUM = 8


def _group_major_geometry(
    mtpr,
    experts_per_rank,
    *,
    world_size=WORLD_SIZE,
    topk=TOPK,
    fixed_slot_dispatch=False,
):
    """Match the runtime group-major capacity and metadata allocation."""
    unit = 32 if fixed_slot_dispatch else 128
    max_tokens_per_expert = world_size * mtpr
    ll_cap = (max_tokens_per_expert + unit - 1) // unit * unit
    if fixed_slot_dispatch:
        num_valid_max = experts_per_rank * ll_cap + 256
    else:
        num_valid_max = world_size * mtpr * topk + (experts_per_rank + 2) * unit
    tile_state_stride = (num_valid_max + 31) // 32
    return ll_cap, num_valid_max, tile_state_stride


def default_jobs(
    mtprs=DEFAULT_MTPRS,
    experts_per_ranks=DEFAULT_EXPERTS_PER_RANKS,
    *,
    quants=DEFAULT_QUANTS,
    world_size=WORLD_SIZE,
    topk=TOPK,
    model_dim=MODEL_DIM,
    inter_dim=INTER_DIM,
    swiglu_limit=SWIGLU_LIMIT,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    shape_suffix = ""
    if (world_size, topk, model_dim, inter_dim) != (
        WORLD_SIZE,
        TOPK,
        MODEL_DIM,
        INTER_DIM,
    ):
        shape_suffix = f"_w{world_size}_k{topk}_d{model_dim}_i{inter_dim}"
    # Activation is baked into the Stage1 epilogue, so a non-default act needs a
    # distinct AOT artifact name (mirrors the runtime kernel-name act suffix).
    act_suffix = "" if act == "silu" else f"_{act}"
    jobs = [
        {
            "kernel_name": (
                f"mega_moe_{quant}_stage{stage}_bundle_mtpr{mtpr}_epr{experts_per_rank}"
                f"_rank{rank}{shape_suffix}{act_suffix}"
            ),
            "quant": quant,
            "stage": stage,
            "mtpr": mtpr,
            "experts_per_rank": experts_per_rank,
            "rank": rank,
            "world_size": world_size,
            "topk": topk,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "swiglu_limit": swiglu_limit,
            "act": act,
            "situ_beta": situ_beta,
            "situ_linear_beta": situ_linear_beta,
        }
        for mtpr in mtprs
        for experts_per_rank in experts_per_ranks
        for quant in quants
        for rank in range(world_size)
        for stage in (1, 2)
    ]
    fused_shape = (
        "a8w4" in quants
        and FUSED_PATH_EXPERTS_PER_RANK in experts_per_ranks
        and (world_size, topk, model_dim, inter_dim, swiglu_limit, act)
        == (WORLD_SIZE, TOPK, MODEL_DIM, INTER_DIM, SWIGLU_LIMIT, "silu")
    )
    if fused_shape:
        jobs += fused_path_jobs(large_m=LARGE_M_MTPR in mtprs)
    return jobs


def fused_path_jobs(*, large_m=True):
    """Small-input runtimes (MTPR 256/512) and the large-input kernel (MTPR 8192) for every rank."""
    common = {
        "quant": "a8w4",
        "experts_per_rank": FUSED_PATH_EXPERTS_PER_RANK,
        "world_size": WORLD_SIZE,
        "topk": TOPK,
        "model_dim": MODEL_DIM,
        "inter_dim": INTER_DIM,
        "swiglu_limit": SWIGLU_LIMIT,
    }
    stages = [("small_m", mtpr) for mtpr in SMALL_M_MTPRS]
    if large_m:
        stages.append(("large_m", LARGE_M_MTPR))
    return [
        {
            "kernel_name": (
                f"mega_moe_a8w4_{stage}_mtpr{mtpr}_epr{FUSED_PATH_EXPERTS_PER_RANK}_rank{rank}"
            ),
            "stage": stage,
            "mtpr": mtpr,
            "rank": rank,
            **common,
        }
        for stage, mtpr in stages
        for rank in range(WORLD_SIZE)
    ]


def production_jobs(
    mtprs=DEFAULT_MTPRS,
    *,
    quants=DEFAULT_QUANTS,
    world_size=WORLD_SIZE,
    swiglu_limit=SWIGLU_LIMIT,
):
    """Every geometry the service ships: V4-Pro (default shape) + Kimi-K3.

    Used by the default build so both profiles are AOT-covered and neither
    triggers an online JIT compile at serving time.
    """
    jobs = default_jobs(
        mtprs, quants=quants, world_size=world_size, swiglu_limit=swiglu_limit
    )
    jobs += default_jobs(
        mtprs,
        K3_EXPERTS_PER_RANKS,
        quants=quants,
        world_size=world_size,
        topk=K3_TOPK,
        model_dim=K3_MODEL_DIM,
        inter_dim=K3_INTER_DIM,
        # SiTUv2 self-saturates -> no hard clamp; keep swiglu_limit at 0 so the
        # kernel name matches the runtime kimi_k3_route (swiglu_limit=0.0).
        swiglu_limit=0.0,
        act=K3_ACT,
        situ_beta=K3_SITU_BETA,
        situ_linear_beta=K3_SITU_LINEAR_BETA,
    )
    return jobs


def _tensor(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="cpu")


def _compile_stage1(
    mtpr,
    experts_per_rank,
    rank,
    plan,
    *,
    a_dtype,
    world_size,
    topk,
    model_dim,
    inter_dim,
    swiglu_limit,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_prepare import (
        preload_mega_moe_prepare,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import (
        compile_mega_moe_stage1_bundle,
    )
    from aiter.ops.flydsl.kernels.mega_moe.quant import _get_launcher

    fuse_cap, _, tile_state_stride = _group_major_geometry(
        mtpr,
        experts_per_rank,
        world_size=world_size,
        topk=topk,
        fixed_slot_dispatch=plan.fixed_slot_dispatch,
    )
    scale_dim = model_dim // 32
    seen_prepare = set()
    prepare_entries = () if plan.fixed_slot_dispatch else plan.entries
    for entry in prepare_entries:
        config = entry.config.stage1
        prepare_blocks = max(
            1, min(config.num_dispatch_cu, (entry.token_bucket + 63) // 64)
        )
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
                fuse_npes=world_size,
                fuse_topk=topk,
                fuse_mtpr=mtpr,
                sort_block_m=config.sort_block_m,
                num_dispatch_cu=config.num_dispatch_cu,
                num_prepare_cu=prepare_blocks,
                num_quant_cu=num_quant_cu,
                quant_cu_capacity=NUM_CU,
                quant_mode=a_dtype,
                model_dim=model_dim,
                payload_chunk_rows=config.payload_chunk_rows,
                tile_state_stride=tile_state_stride,
            )

    # The production E2E path quantizes inside prepare.  Keep the public
    # standalone quantize() launcher in the same AOT profile as well because
    # the validation/per-stage benchmark invokes it to isolate Stage1 timing.
    quant_rows = 2
    quant_groups = quant_rows * scale_dim
    output_dtype = torch.uint8 if a_dtype == "fp4" else torch.float8_e4m3fn
    output_dim = model_dim // 2 if a_dtype == "fp4" else model_dim
    _get_launcher(model_dim, a_dtype)(
        _tensor((quant_rows, model_dim), torch.bfloat16),
        _tensor((quant_rows, output_dim), output_dtype),
        _tensor((quant_rows, scale_dim), torch.uint8),
        quant_rows,
        (quant_groups + 63) // 64,
        fx.Stream(None),
    )

    launch = compile_mega_moe_stage1_bundle(
        model_dim=model_dim,
        inter_dim=inter_dim,
        rank=rank,
        experts_per_rank=experts_per_rank,
        fuse_npes=world_size,
        fuse_topk=topk,
        fuse_cap=fuse_cap,
        fuse_mtpr=mtpr,
        fuse_scale_dim=model_dim // 32,
        fixed_slot_dispatch=plan.fixed_slot_dispatch,
        num_cu=NUM_CU,
        tile_state_stride=tile_state_stride,
        variants=plan.stage1_variants,
        swiglu_limit=swiglu_limit,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        a_dtype=a_dtype,
        out_dtype=a_dtype,
    )
    launch(
        _tensor((1, inter_dim // 2 if a_dtype == "fp4" else inter_dim), output_dtype),
        _tensor((1, output_dim), output_dtype),
        # Keep every non-unit dimension larger than one so FlyDSL records the
        # same contiguous-stride signature as the runtime weight tensor.  An
        # all-ones placeholder makes the first dimension look unit-strided and
        # produces an AOT cache key that the real tensor can never reuse.
        _tensor((2, 2, 2), torch.uint8),
        (
            _tensor((1, model_dim // 32), torch.uint8)
            if a_dtype == "fp4"
            else _tensor((1, model_dim // 128), torch.int32)
        ),
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


def _compile_stage2(
    mtpr,
    experts_per_rank,
    rank,
    plan,
    *,
    a_dtype,
    world_size,
    topk,
    model_dim,
    inter_dim,
):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
        preload_mega_moe_stage2,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2_aligned_pair import (
        ALIGNED_PAIR_SCATTER_VEC,
        preload_mega_moe_stage2_aligned_pair,
    )

    _, num_valid_max, _ = _group_major_geometry(
        mtpr,
        experts_per_rank,
        world_size=world_size,
        topk=topk,
        fixed_slot_dispatch=plan.fixed_slot_dispatch,
    )
    for key in plan.stage2_variants:
        stage2 = key.config
        row_bytes = (
            model_dim + model_dim // 32
            if key.p2p_quant == "fp8_blockwise_1x32"
            else model_dim * 2
        )
        common = {
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "experts": experts_per_rank,
            "topk": topk,
            "rank": rank,
            "npes": world_size,
            "max_tok": mtpr,
            "recv_cap": world_size * mtpr,
            "comb_inp_nbytes": mtpr * topk * row_bytes,
            "HIDDEN_MAX": model_dim,
            "INTER_MAX": inter_dim,
            "cu_num": NUM_CU,
            "p2p_quant_type": key.p2p_quant,
            "fixed_slot_dispatch": plan.fixed_slot_dispatch,
            "a_dtype": a_dtype,
        }
        residual = (
            replace(stage2, skew_cu=stage2.persist_cu)
            if stage2.aligned_pair
            else stage2
        )
        preload_mega_moe_stage2(
            *([fx.Int64(0)] * 15),
            num_valid_max,
            fx.Int32(inter_dim),
            fx.Int32(model_dim),
            fx.Stream(None),
            BM=residual.block_m,
            SBM=key.sbm,
            BN=residual.block_n,
            BK=residual.block_k,
            use_nt=residual.use_nt,
            g2_bhoist=residual.b_hoist,
            g2_b2stage=residual.b2stage,
            g2_ascale_pf=residual.ascale_prefetch,
            g2_spart=residual.spatial_partition,
            g2_deep_a_pipeline=residual.deep_a_pipeline,
            persist=residual.persist,
            persist_cu=residual.persist_cu,
            persist_strided=residual.persist_strided,
            skew_cu=residual.skew_cu,
            g2_bf16_lds=residual.bf16_lds,
            runtime_pair_skip=stage2.aligned_pair,
            scatter_vec=ALIGNED_PAIR_SCATTER_VEC if stage2.aligned_pair else 8,
            **common,
        )
        if not stage2.aligned_pair:
            continue
        preload_mega_moe_stage2_aligned_pair(
            *([fx.Int64(0)] * 11),
            num_valid_max,
            fx.Int32(inter_dim),
            fx.Int32(model_dim),
            fx.Stream(None),
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts_per_rank,
            topk=topk,
            rank=rank,
            npes=world_size,
            max_tok=mtpr,
            recv_cap=world_size * mtpr,
            comb_inp_nbytes=mtpr * topk * row_bytes,
            BM=stage2.pair_block_m,
            SBM=key.sbm,
            BN=stage2.pair_block_n,
            BK=stage2.block_k,
            INTER_MAX=inter_dim,
            use_nt=stage2.use_nt,
            cu_num=stage2.pair_cu,
            g2_bhoist=stage2.b_hoist,
            g2_b2stage=stage2.b2stage,
            g2_ascale_pf=stage2.ascale_prefetch,
            g2_deep_a_pipeline=stage2.deep_a_pipeline,
            a_dtype=a_dtype,
        )

    # Stage2's production bundle includes the terminal fused combine kernels.
    # Compile them here as well; otherwise a clean AOT-only service still falls
    # back to JIT after all GEMM2 variants have loaded successfully.
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.flydsl.kernels.communication_ops_utils import (
        GeometryTuningTable,
    )
    from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_kernel import (
        make_combine_jit,
    )

    tuning = GeometryTuningTable()
    tuning_path = Path(AITER_CONFIGS.AITER_CONFIG_DISPATCH_COMBINE_INTRANODE_FILE)
    if tuning_path.is_file():
        tuning = GeometryTuningTable.from_tuning_file(
            tuning_path,
            ep_size=world_size,
            gfx="gfx950",
            gpu_model="mi355x",
            dtype="fp8_ocp",
            hidden_dim=model_dim,
            zero_copy=False,
            topk=topk,
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
            npes=world_size,
            experts_per_token=topk,
            hidden_dim=model_dim,
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
            max_recv=world_size,
        )
        launch(
            *([fx.Int64(0)] * 18),
            fx.Int32(entry.token_bucket),
            fx.Stream(None),
        )


def _main_stage_tensors(model_dim, inter_dim):
    """Stage1 placeholders with the runtime tensors' dtype and rank (see _compile_stage1)."""
    return (
        _tensor((1, inter_dim), torch.float8_e4m3fn),
        _tensor((1, model_dim), torch.float8_e4m3fn),
        _tensor((2, 2, 2), torch.uint8),
        _tensor((1, model_dim // 128), torch.int32),
        _tensor((2, 2), torch.uint8),
        _tensor((1,), torch.int32),
        _tensor((1,), torch.int32),
    )


def _compile_small_m(
    mtpr, rank, *, world_size, topk, model_dim, inter_dim, swiglu_limit
):
    """Every config a small-input runtime of this MTPR selects (one per token bucket)."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm import TOKEN_BUCKETS
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import (
        _COMBINE_REDUCERS,
        run_mega_moe_stage1,
        small_m_config,
    )

    epr = FUSED_PATH_EXPERTS_PER_RANK
    fuse_cap, num_valid_max, _ = _group_major_geometry(
        mtpr, epr, world_size=world_size, topk=topk, fixed_slot_dispatch=True
    )
    seen = set()
    for bucket in (b for b in TOKEN_BUCKETS if b <= mtpr):
        config = small_m_config(
            bucket, mtpr, experts_per_rank=epr, model_dim=model_dim, inter_dim=inter_dim
        )
        stage1, stage2 = config.stage1, config.stage2
        if (stage1, stage2) in seen:
            continue
        seen.add((stage1, stage2))
        run_mega_moe_stage1(
            *_main_stage_tensors(model_dim, inter_dim),
            _tensor((1,), torch.uint8),
            fx.Int32(num_valid_max),
            fx.Int64(0),
            fx.Int32(1),
            *([fx.Int64(0)] * 6),
            fx.Stream(None),
            model_dim=model_dim,
            inter_dim=inter_dim,
            rank=rank,
            experts_per_rank=epr,
            fuse_npes=world_size,
            fuse_topk=topk,
            fuse_cap=fuse_cap,
            fuse_mtpr=mtpr,
            fuse_scale_dim=model_dim // 32,
            fused=(
                stage2.block_n,
                stage2.block_k,
                world_size * mtpr,
                _COMBINE_REDUCERS,
            ),
            num_cu=NUM_CU,
            sort_block_m=stage1.sort_block_m,
            tile_n=stage1.tile_n,
            tile_k=stage1.tile_k,
            num_waves=stage1.num_waves,
            grid_mult=min(stage1.grid_mult, 2),
            pipe_weights=stage1.pipe_weights,
            mfma_amajor=stage1.mfma_amajor,
            swizzle_a=stage1.swizzle_a,
            async_a_copy=stage1.async_a_copy,
            num_dispatch_cu=stage1.num_dispatch_cu,
            use_tile_resource=stage1.use_tile_resource,
            waves_per_eu_hint=stage1.waves_per_eu_hint,
            b_nt=stage1.b_nt,
            swiglu_limit=swiglu_limit,
        )


def _compile_large_m(rank, *, world_size, topk, model_dim, inter_dim, swiglu_limit):
    """The persistent large-input kernel and its quant kernel."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm import (
        select_mega_moe_config,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import (
        LARGE_M_G2_CHUNK,
        large_m_kernel_specs,
        run_mega_moe_stage1,
    )
    from aiter.ops.flydsl.kernels.mega_moe.quant import _get_launcher

    mtpr, epr = LARGE_M_MTPR, FUSED_PATH_EXPERTS_PER_RANK
    fuse_cap, num_valid_max, _ = _group_major_geometry(
        mtpr, epr, world_size=world_size, topk=topk
    )
    config = select_mega_moe_config(
        mtpr, mtpr, experts_per_rank=epr, model_dim=model_dim, inter_dim=inter_dim
    )
    stage1 = config.stage1
    s2_spec, c_spec = large_m_kernel_specs(
        config,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
        rank=rank,
        npes=world_size,
        mtpr=mtpr,
        experts_per_rank=epr,
        num_valid_max=num_valid_max,
    )
    scale_dim = model_dim // 32
    _get_launcher(model_dim, "fp8")(
        _tensor((2, model_dim), torch.bfloat16),
        _tensor((2, model_dim), torch.float8_e4m3fn),
        _tensor((2, scale_dim), torch.uint8),
        2,
        (2 * scale_dim + 63) // 64,
        fx.Stream(None),
    )
    run_mega_moe_stage1(
        *_main_stage_tensors(model_dim, inter_dim),
        _tensor((2,), torch.int32),
        _tensor((1,), torch.uint8),
        fx.Int32(num_valid_max),
        fx.Int64(0),
        fx.Int32(1),
        *([fx.Int64(0)] * 6),
        fx.Stream(None),
        model_dim=model_dim,
        inter_dim=inter_dim,
        rank=rank,
        experts_per_rank=epr,
        fuse_npes=world_size,
        fuse_topk=topk,
        fuse_cap=fuse_cap,
        fuse_mtpr=mtpr,
        fuse_scale_dim=scale_dim,
        num_cu=NUM_CU,
        fused_stage2=s2_spec,
        fused_args=(0,) * 14,
        fused_combine=c_spec,
        fused_combine_args=(0, 0, 0, 0, 0, 1, model_dim + scale_dim),
        sort_block_m=stage1.sort_block_m,
        tile_n=stage1.tile_n,
        tile_k=stage1.tile_k,
        num_waves=stage1.num_waves,
        grid_mult=stage1.grid_mult,
        pipe_weights=stage1.pipe_weights,
        mfma_amajor=stage1.mfma_amajor,
        swizzle_a=stage1.swizzle_a,
        async_a_copy=stage1.async_a_copy,
        num_dispatch_cu=stage1.num_dispatch_cu,
        use_tile_resource=stage1.use_tile_resource,
        waves_per_eu_hint=stage1.waves_per_eu_hint,
        b_nt=stage1.b_nt,
        work_shards=stage1.work_shards,
        payload_chunk_rows=stage1.payload_chunk_rows,
        swiglu_limit=swiglu_limit,
        fused_g2_chunk=LARGE_M_G2_CHUNK,
    )


def compile_one_config(**job):
    result = {**job, "compile_time": None}
    started = time.time()
    try:
        world_size = job.get("world_size", WORLD_SIZE)
        topk = job.get("topk", TOPK)
        model_dim = job.get("model_dim", MODEL_DIM)
        inter_dim = job.get("inter_dim", INTER_DIM)
        swiglu_limit = job.get("swiglu_limit", SWIGLU_LIMIT)
        act = job.get("act", "silu")
        situ_beta = job.get("situ_beta", 1.0)
        situ_linear_beta = job.get("situ_linear_beta", 1.0)
        a_dtype = "fp4" if job["quant"] == "a4w4" else "fp8"
        if job["stage"] in ("small_m", "large_m"):
            shape = {
                "world_size": world_size,
                "topk": topk,
                "model_dim": model_dim,
                "inter_dim": inter_dim,
                "swiglu_limit": swiglu_limit,
            }
            with compile_only_env(), override_env("FLYDSL_GPU_ARCH", "gfx950"):
                if job["stage"] == "small_m":
                    _compile_small_m(job["mtpr"], job["rank"], **shape)
                else:
                    _compile_large_m(job["rank"], **shape)
            result["compile_time"] = time.time() - started
            return result
        plan = build_mega_moe_bundle_plan(
            job["mtpr"],
            a_dtype=a_dtype,
            experts_per_rank=job["experts_per_rank"],
            model_dim=model_dim,
            inter_dim=inter_dim,
            world_size=world_size,
        )
        with compile_only_env(), override_env("FLYDSL_GPU_ARCH", "gfx950"):
            if job["stage"] == 1:
                _compile_stage1(
                    job["mtpr"],
                    job["experts_per_rank"],
                    job["rank"],
                    plan,
                    a_dtype=a_dtype,
                    world_size=world_size,
                    topk=topk,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                    swiglu_limit=swiglu_limit,
                    act=act,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
            else:
                _compile_stage2(
                    job["mtpr"],
                    job["experts_per_rank"],
                    job["rank"],
                    plan,
                    a_dtype=a_dtype,
                    world_size=world_size,
                    topk=topk,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                )
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
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--model-dim", type=int, default=MODEL_DIM)
    parser.add_argument("--inter-dim", type=int, default=INTER_DIM)
    parser.add_argument("--swiglu-limit", type=float, default=SWIGLU_LIMIT)
    parser.add_argument("--act", default="silu", choices=("silu", "situv2"))
    parser.add_argument("--situ-beta", type=float, default=1.0)
    parser.add_argument("--situ-linear-beta", type=float, default=1.0)
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=DEFAULT_QUANTS,
        default=list(DEFAULT_QUANTS),
    )
    args = parser.parse_args()
    default_shape = (
        list(args.experts_per_rank) == list(DEFAULT_EXPERTS_PER_RANKS)
        and args.topk == TOPK
        and args.model_dim == MODEL_DIM
        and args.inter_dim == INTER_DIM
    )
    if default_shape:
        # Default build ships both V4-Pro and Kimi-K3 so neither JITs online.
        jobs = production_jobs(
            tuple(args.mtpr),
            quants=tuple(args.quant),
            world_size=args.world_size,
            swiglu_limit=args.swiglu_limit,
        )
    else:
        jobs = default_jobs(
            tuple(args.mtpr),
            tuple(args.experts_per_rank),
            quants=tuple(args.quant),
            world_size=args.world_size,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            swiglu_limit=args.swiglu_limit,
            act=args.act,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
        )
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failed} ok, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

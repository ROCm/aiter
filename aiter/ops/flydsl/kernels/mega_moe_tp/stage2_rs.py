# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEMM2 + weighted top-k reduce + ReduceScatter in one kernel launch."""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from ..mxfp4_gemm_common import _udiv, global_typed_ptr
from ..mxmoe_dispatcher import (
    _active_m_blocks_upper_bound,
    _spart_output_tile_index,
    compile_gemm2_a4w4_port,
)
from ..tensor_shim import _run_compiled as run_compiled
from .reduce_scatter import MAX_SERVICE_BLOCKS, RS_UNIT_ELEMS
from .rs_tail import emit_rs_tail, read_epoch, rs_tail_slots

__all__ = ["compile_stage2_rs", "run_stage2_rs", "stage2_rs_supported"]

_BLOCK = 256
_SERVICE_BLOCKS = min(
    MAX_SERVICE_BLOCKS, int(os.environ.get("AITER_TP_STAGE2_RS_SERVICE", "128"))
)


@functools.cache
def compile_stage2_rs(
    tp_size: int,
    model_dim: int,
    *,
    BM: int,
    BN: int,
    BK: int,
    use_nt: bool,
    HIDDEN_MAX: int,
    INTER_MAX: int,
    a_dtype: str,
    b_dtype: str,
    SBM: int | None,
    g2_bhoist: bool | None = None,
    g2_ascale_pf: bool | None = None,
    g2_spart: int | None = None,
    g2_bf16_lds: bool | None = None,
    g2_kstatic: bool = False,
    service_blocks: int = _SERVICE_BLOCKS,
):
    """Build the fused GEMM2 + ReduceScatter launcher for one tuned GEMM2 row."""
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    if model_dim % RS_UNIT_ELEMS:
        raise ValueError(
            f"model_dim must be a multiple of {RS_UNIT_ELEMS}, got {model_dim}"
        )
    if service_blocks <= 0:
        raise ValueError(f"service_blocks must be positive, got {service_blocks}")

    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0

    tail_slots = rs_tail_slots(tp_size)

    def compose(*, module_name, emit_gemm2_tile, shared_storage, **_extra):
        name = f"{module_name}_tp{tp_size}_rs_h{model_dim}_sv{service_blocks}"

        @flyc.kernel(name=name, known_block_size=[_BLOCK, 1, 1])
        def stage2_rs_kernel(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            arg_bias: fx.Int64,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_out: fx.Int64,
            arg_desc: fx.Int64,
            i32_rank: fx.Int32,
            i32_rows: fx.Int32,
        ):
            tx_i32 = fx.Int32(gpu.thread_id("x"))
            bx_i32 = fx.Int32(gpu.block_id("x"))
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            grid_nb = fx.Int32(gpu.grid_dim.x)
            lds = fx.SharedAllocator().allocate(shared_storage).peek()

            epoch_addr, epoch = read_epoch(arg_desc, tail_slots)

            num_n_blocks = fx.Int32(fx.Uint32(i32_hidden) // fx.Uint32(BN))
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * num_n_blocks
            if bx_i32 < bound:
                if const_expr(g2_spart > 0):
                    m_block_idx, n_block_idx = _spart_output_tile_index(
                        bx_i32,
                        total_m_blocks,
                        num_n_blocks,
                        g2_group_num,
                        g2_m01,
                    )
                else:
                    m_block_idx = _udiv(bx_i32, num_n_blocks)
                    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
                emit_gemm2_tile(
                    arg_aq,
                    arg_ascale,
                    arg_bq,
                    arg_bscale,
                    arg_eids,
                    arg_stids,
                    arg_sweights,
                    arg_bias,
                    arg_out,
                    m_block_idx,
                    n_block_idx,
                    lane,
                    wave,
                    i32_M,
                    i32_max_m_blocks,
                    i32_inter,
                    i32_hidden,
                    lds,
                )

            emit_rs_tail(
                arg_desc,
                i32_rank,
                i32_rows,
                epoch_addr,
                epoch,
                bx_i32,
                grid_nb,
                tx_i32,
                tp_size=tp_size,
                model_dim=model_dim,
                block=_BLOCK,
                service_blocks=service_blocks,
            )


        @flyc.jit
        def launch_stage2_rs(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            arg_bias: fx.Int64,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_grid_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_out: fx.Int64,
            arg_desc: fx.Int64,
            i32_rank: fx.Int32,
            i32_rows: fx.Int32,
            stream: fx.Stream,
        ):
            num_n_blocks = fx.Int32(fx.Uint32(i32_hidden) // fx.Uint32(BN))
            grid_x = i32_grid_blocks * num_n_blocks
            stage2_rs_kernel(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_bias,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                arg_out,
                arg_desc,
                i32_rank,
                i32_rows,
            ).launch(grid=(grid_x, 1, 1), block=(_BLOCK, 1, 1), stream=stream)

        launch_stage2_rs.block = _BLOCK
        launch_stage2_rs.service_blocks = service_blocks
        return launch_stage2_rs

    return compile_gemm2_a4w4_port(
        BM=BM,
        BN=BN,
        BK=BK,
        use_nt=use_nt,
        HIDDEN_MAX=HIDDEN_MAX,
        epilog="atomic",
        INTER_MAX=INTER_MAX,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        topk=1,
        SBM=SBM,
        persist=False,
        g2_bhoist=g2_bhoist,
        g2_ascale_pf=g2_ascale_pf,
        g2_spart=g2_spart,
        g2_bf16_lds=g2_bf16_lds,
        g2_kstatic=g2_kstatic,
        out_dtype="bf16",
        enable_bias=False,
        _composition=compose,
    )


def _as_u8(tensor):
    """The uint8 view the GEMM2 ABI wants for every packed operand."""
    if tensor.element_size() == 1 and tensor.dtype != torch.uint8:
        return tensor.view(torch.uint8)
    return tensor


def stage2_rs_supported(kernel_cfg) -> bool:
    """Whether a parsed ``flydsl_moe2_layout_*`` row can carry the fused tail."""
    return (
        kernel_cfg is not None
        and kernel_cfg.get("epilog") in ("atomic", "reduce")
        and not kernel_cfg.get("persist")
    )


def run_stage2_rs(
    *,
    inter_states,
    a2_scale,
    w2,
    w2_scale,
    sorted_expert_ids,
    num_valid_ids,
    sorted_token_ids,
    sorted_weights,
    partial,
    output,
    desc_ptr,
    rank,
    tp_size,
    local_rows,
    M_logical,
    NE,
    model_dim,
    inter_dim,
    topk,
    kernel_cfg,
    block_m=None,
    stream=None,
):
    """Host side of the fused GEMM2 + ReduceScatter."""
    BM = kernel_cfg["tile_m"]
    BN = kernel_cfg["tile_n"]
    BK = kernel_cfg["tile_k"]
    SBM = kernel_cfg["sort_block_m"] or (int(block_m) if block_m else BM)
    if model_dim % BN:
        raise ValueError(f"model_dim {model_dim} must be a multiple of BN {BN}")
    if inter_dim % BK:
        raise ValueError(f"inter_dim {inter_dim} must be a multiple of BK {BK}")

    kstatic = os.environ.get("MXFP4_G2_KSTATIC", "1") == "1"
    launch = compile_stage2_rs(
        int(tp_size),
        int(model_dim),
        BM=BM,
        BN=BN,
        BK=BK,
        use_nt=kernel_cfg["use_nt"],
        HIDDEN_MAX=8192,
        INTER_MAX=inter_dim if kstatic else 8192,
        a_dtype=kernel_cfg["a_dtype"],
        b_dtype=kernel_cfg["b_dtype"],
        SBM=SBM,
        g2_bf16_lds=kernel_cfg["bf16_lds"],
        g2_spart=kernel_cfg["spart"],
        g2_kstatic=kstatic,
        service_blocks=_SERVICE_BLOCKS,
    )

    max_sorted = inter_states.shape[0]
    max_m_blocks = (max_sorted + BM - 1) // BM
    grid_blocks = min(
        max_m_blocks,
        _active_m_blocks_upper_bound(M_logical, topk, NE, BM, SBM),
    )
    run_compiled(
        launch,
        _as_u8(inter_states).data_ptr(),
        _as_u8(a2_scale).data_ptr(),
        _as_u8(w2).data_ptr(),
        _as_u8(w2_scale).data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        partial.data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid_blocks),
        int(inter_dim),
        int(model_dim),
        partial.data_ptr(),
        int(desc_ptr),
        int(rank),
        int(local_rows),
        stream if stream is not None else torch.cuda.current_stream(),
    )
    return output[:local_rows]

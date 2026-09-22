# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEMM1 + GEMM2 + ReduceScatter in one kernel launch."""


import functools
import hashlib
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from ..mxfp4_gemm1 import compile_gemm1_a4w4_port
from ..mxfp4_gemm_common import _udiv, _umod, global_typed_ptr
from .. import communication_ops_utils as comm
from .p2p import desc_slot
from ..mxmoe_dispatcher import compile_gemm2_a4w4_port
from ..tensor_shim import _run_compiled as run_compiled, buf_copy_atom, ptr_buf_tensor
from .reduce_scatter import MAX_SERVICE_BLOCKS, RS_UNIT_ELEMS
from .rs_tail import (
    emit_phase_barrier,
    emit_rs_tail,
    read_epoch,
    rs_service_low as _rs_service_low,
    rs_tail_slots,
)

__all__ = [
    "compile_stage12_rs",
    "hosts_reduce_as_atomic",
    "run_stage12_rs",
    "stage12_supported",
]

_BLOCK = 256
_SERVICE_BLOCKS = min(
    MAX_SERVICE_BLOCKS, int(os.environ.get("AITER_TP_STAGE12_RS_SERVICE", "128"))
)
_GRID_CU = int(os.environ.get("AITER_TP_STAGE12_GRID", "0"))
_AG_GRID_CAP = int(os.environ.get("AITER_TP_MEGA_AG_GRID_CAP", "0"))
_WAVES_PER_EU = int(os.environ.get("AITER_TP_MEGA_WAVES_PER_EU", "2"))
_ROLL_ENV = os.environ.get("AITER_TP_MEGA_ROLL", "auto")
_ROLL_MIN_NB = int(os.environ.get("AITER_TP_MEGA_ROLL_MIN_NB", "16"))
_ASCALE_SHUFFLE = os.environ.get("AITER_TP_MEGA_ASCALE_SHUFFLE", "1") == "1"


def _roll_enabled(g2_n_blocks: int) -> bool:
    if _ROLL_ENV == "auto":
        return int(g2_n_blocks) >= _ROLL_MIN_NB
    return _ROLL_ENV == "1"


def _mx_scale_shuffle_idx(scaleN_pad: int, x, y):
    """Byte offset of scale column ``y`` of sorted row ``x`` in GEMM1's layout."""
    row_term = (
        _udiv(x, fx.Int32(32)) * fx.Int32(scaleN_pad * 32)
        + _umod(x, fx.Int32(16)) * fx.Int32(4)
        + _udiv(_umod(x, fx.Int32(32)), fx.Int32(16))
    )
    if isinstance(y, int):
        return row_term + fx.Int32((y // 8) * 256 + (y % 4) * 64 + (y % 8) // 4 * 2)
    return (
        row_term
        + _udiv(y, fx.Int32(8)) * fx.Int32(256)
        + _umod(y, fx.Int32(4)) * fx.Int32(64)
        + _udiv(_umod(y, fx.Int32(8)), fx.Int32(4)) * fx.Int32(2)
    )


@flyc.jit
def emit_ascale_shuffle(
    arg_scale_in,
    arg_scale_out,
    arg_stids,
    arg_num_valid,
    i32_ntok,
    row0,
    row_stride,
    tid,
    stride,
    total_sorted,
    *,
    scale_per_row: int,
    scaleN_pad: int,
):
    """Reorder the gathered E8M0 scales into the layout GEMM1 reads."""
    src = global_typed_ptr(arg_scale_in, T.i8, align=1)
    dst = global_typed_ptr(arg_scale_out, T.i8, align=1)
    num_valid = global_typed_ptr(arg_num_valid, T.i32)[0]
    _nw = stride // fx.Int32(64)
    _wv = tid // fx.Int32(64)
    _ln = _umod(tid, fx.Int32(64))
    for raw in range(row0 + _wv * row_stride, total_sorted, row_stride * _nw):
        row = fx.Int32(raw)
        if row < num_valid:
            info = global_typed_ptr(arg_stids, T.i32)[row]
            token = info & fx.Int32(0xFFFFFF)
            if token < i32_ntok:
                base = token * fx.Int32(scale_per_row)
                for col in range(_ln, fx.Int32(scale_per_row), fx.Int32(64)):
                    c = fx.Int32(col)
                    dst[_mx_scale_shuffle_idx(scaleN_pad, row, c)] = src[base + c]


def hosts_reduce_as_atomic() -> bool:
    """A ``reduce``-tuned row always runs its GEMM2 atomically in here."""
    return True


def stage12_supported(g1_cfg, g2_cfg, model_dim: int) -> bool:
    """Whether this tuned (GEMM1, GEMM2) pair can share one kernel."""
    if g1_cfg is None or g2_cfg is None:
        return False
    if model_dim % RS_UNIT_ELEMS:
        return False
    if g1_cfg.get("a_dtype") != "fp4" or g1_cfg.get("out_dtype") != "fp4":
        return False
    if int(g1_cfg.get("num_waves", 4)) * int(g1_cfg.get("k_wave", 1)) * 64 != _BLOCK:
        return False
    if int(g1_cfg.get("BM", 0)) == 16:
        return False
    if g2_cfg.get("epilog") not in ("atomic", "reduce") or g2_cfg.get("persist"):
        return False
    if int(g2_cfg.get("tile_m", 0)) != int(g1_cfg.get("BM", -1)):
        return False
    if g2_cfg.get("spart"):
        return False
    return True


@functools.cache
def compile_stage12_rs(
    tp_size: int,
    model_dim: int,
    *,
    g1_BM: int,
    g1_use_nt: bool,
    g1_inline_quant: bool,
    g1_act: str,
    g1_situ_beta: float,
    g1_situ_linear_beta: float,
    g1_swiglu_limit: float,
    g1_native_scale_layout: bool,
    g1_interleave: bool,
    g1_xcd_swizzle: int,
    g1_num_waves: int,
    g1_k_wave: int,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    g2_BM: int,
    g2_BN: int,
    g2_BK: int,
    g2_use_nt: bool,
    g2_SBM: int,
    g2_spart: int | None,
    g2_bf16_lds: bool | None,
    g2_kstatic: bool,
    HIDDEN_MAX: int,
    INTER_MAX: int,
    a_dtype: str,
    b_dtype: str,
    topk: int = 0,
    fuse_ag: bool = False,
    g2_epilog: str = "atomic",
    waves_per_eu: int = 0,
    service_blocks: int = _SERVICE_BLOCKS,
):
    """Build the fused GEMM1 + GEMM2 + ReduceScatter launcher for one row pair."""
    _config = dict(locals())
    g2_tag = "_red" if g2_epilog == "reduce" else ""
    if g2_epilog == "reduce":
        g2_epilog = "atomic"
        g2_tag = "_reda"
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")

    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    tail_slots = rs_tail_slots(tp_size)

    g1: dict = {}

    def g1_compose(**hook):
        g1.update(hook)
        return _G1Handle()

    compile_gemm1_a4w4_port(
        BM=g1_BM,
        use_nt=g1_use_nt,
        inline_quant=g1_inline_quant,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        interleave=g1_interleave,
        xcd_swizzle=g1_xcd_swizzle,
        a_dtype="fp4",
        out_dtype="fp4",
        act=g1_act,
        situ_beta=g1_situ_beta,
        situ_linear_beta=g1_situ_linear_beta,
        swiglu_limit=g1_swiglu_limit,
        native_scale_layout=g1_native_scale_layout,
        num_waves=g1_num_waves,
        k_wave=g1_k_wave,
        _composition=g1_compose,
    )
    if int(g1["block_threads"]) != _BLOCK:
        raise ValueError(
            f"GEMM1 wants {g1['block_threads']} threads but GEMM2 is fixed at "
            f"{_BLOCK}; this pair cannot share a kernel"
        )
    emit_gemm1_tile = g1["emit_gemm1_tile"]
    g1_n_blocks = int(g1["n_blocks"])
    g1_lds_bytes = int(g1["lds_bytes"])

    ag: dict = {}
    if fuse_ag:
        from .allgather_quant_push import (
            compile_allgather_quant_push,
            quant_push_supported,
        )

        if not quant_push_supported(model_dim, topk):
            raise ValueError(
                f"model_dim={model_dim} topk={topk} has no fused quant push, so "
                "the AllGather cannot move into this kernel"
            )
        compile_allgather_quant_push(
            tp_size,
            model_dim,
            topk,
            block=_BLOCK,
            _composition=lambda **hook: ag.update(hook),
        )

    emit_quant_payload_push = ag.get("emit_quant_payload_push")
    emit_ag_barrier = ag.get("emit_ag_barrier")
    emit_ag_gate = ag.get("emit_ag_gate")
    push_ctas = 0
    if fuse_ag:
        from aiter.jit.utils.chip_info import get_cu_num as _cu

        push_ctas = int(_cu())
    if fuse_ag:
        from .allgather_push import AG_DESC_DONE, AG_DESC_EPOCH
        from .p2p import desc_size as _p2p_desc_size

        ag_done_index = _p2p_desc_size(tp_size, 4) + AG_DESC_DONE
        assert AG_DESC_EPOCH == AG_DESC_DONE - 1
    else:
        ag_done_index = 0

    def g2_compose(*, module_name, emit_gemm2_tile, shared_storage, lds_bytes, **_):
        merged_lds = max(int(lds_bytes), g1_lds_bytes, 2 * g2_BM * 4)
        G2_N_BLOCKS = int(model_dim) // int(g2_BN)
        _ROLL = _roll_enabled(G2_N_BLOCKS)

        sig = hashlib.sha256(
            repr(
                (
                    _config, _ASCALE_SHUFFLE, _ROLL, _rs_service_low(),
                )
            ).encode()
        ).hexdigest()[:12]
        name = (
            f"mega_moe_tp_{'mega' if fuse_ag else 'stage12'}_rs_tp{tp_size}"
            f"_h{model_dim}_g1bm{g1_BM}_g2bm{g2_BM}x{g2_BN}"
            f"{g2_tag}_sv{service_blocks}_{sig}"
        )
        if os.environ.get("AITER_TP_MEGA_ECHO", "0") == "1":
            print(
                f"[ECHO] fuse_ag={fuse_ag} "
                f"g1_n_blocks={g1_n_blocks} G2_N_BLOCKS={G2_N_BLOCKS} "
                f"_GRID_CU={_GRID_CU} g1_lds={g1_lds_bytes} g2_lds={int(lds_bytes)} "
                f"merged_lds={merged_lds} g1_BM={g1_BM} g2_BM={g2_BM} name={name}",
                flush=True,
            )

        @fx.struct
        class MergedStorage:
            buf: fx.Array[Int8, merged_lds, 16]
            task: fx.Array[fx.Int32, 4, 16]

        @flyc.kernel(name=name, known_block_size=[_BLOCK, 1, 1])
        def stage12_kernel(
            arg_hidden: fx.Int64,
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1_scale: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2_scale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            arg_mind: fx.Int64,
            arg_bias1: fx.Int64,
            arg_bias2: fx.Int64,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_out: fx.Int64,
            i32_ntok: fx.Int32,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_desc: fx.Int64,
            i32_rank: fx.Int32,
            i32_rows: fx.Int32,
            arg_ag_desc: fx.Int64,
            arg_ascale_raw: fx.Int64,
            i32_max_sorted: fx.Int32,
            arg_tk_ids: fx.Int64,
            arg_tk_weights: fx.Int64,
            arg_num_valid: fx.Int64,
            arg_dbg: fx.Int64,
            arg_taskq: fx.Int64,
        ):
            tx_i32 = fx.Int32(gpu.thread_id("x"))
            bx_i32 = fx.Int32(gpu.block_id("x"))
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            grid_nb = fx.Int32(gpu.grid_dim.x)
            _BSTEP = 2
            lds = fx.SharedAllocator().allocate(MergedStorage).peek()
            lds_raw = lds.buf.ptr
            epoch_addr, epoch = read_epoch(arg_desc, tail_slots)

            if fuse_ag:
                ag_epoch0 = epoch
                n_push = fx.min(fx.Int32(push_ctas), grid_nb)
                if bx_i32 < n_push:
                    emit_quant_payload_push(
                        arg_ag_desc,
                        i32_rank,
                        i32_rows,
                        bx_i32,
                        bx_i32 * fx.Int32(_BLOCK) + tx_i32,
                        n_push * fx.Int32(_BLOCK),
                    )
                    emit_ag_barrier(
                        arg_ag_desc,
                        i32_rank,
                        n_push,
                        tx_i32,
                        entry_epoch=ag_epoch0,
                    )
                emit_ag_gate(arg_ag_desc, ag_epoch0, tx_i32)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]

            g1_total_m = _udiv(cumsum0, g1_BM)
            _NXCD = 8
            _xq = _udiv(g1_total_m, _NXCD)
            _xr = _umod(g1_total_m, _NXCD)

            def _g1_flat_tile(pid, bound):
                """XCD-round-robin over the flattened (m, n) GEMM1 tile space."""
                if const_expr(g1_xcd_swizzle <= 0):
                    return pid
                xq = _udiv(bound, _NXCD)
                xr = _umod(bound, _NXCD)
                xc = _umod(pid, _NXCD)
                wgid = xc * xq + fx.min(xc, xr) + _udiv(pid, _NXCD)
                ng = fx.Int32(g1_xcd_swizzle * g1_n_blocks)
                group_id = wgid // ng
                first_pid_m = group_id * fx.Int32(g1_xcd_swizzle)
                remaining_m = g1_total_m - first_pid_m
                group_size_m = fx.min(remaining_m, fx.Int32(g1_xcd_swizzle))
                wig = wgid % ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(g1_n_blocks) + n_block

            def _m_block(pid):
                """Spread consecutive row blocks across XCDs, as GEMM1 does."""
                if const_expr(g1_xcd_swizzle <= 0):
                    return pid
                xc = _umod(pid, _NXCD)
                return xc * _xq + fx.min(xc, _xr) + _udiv(pid, _NXCD)

            if _GRID_CU > 0:
                if _ASCALE_SHUFFLE:
                    emit_ascale_shuffle(
                        arg_ascale_raw,
                        arg_ascale,
                        arg_stids,
                        arg_cumsum,
                        i32_ntok,
                        bx_i32,
                        grid_nb,
                        tx_i32,
                        fx.Int32(_BLOCK),
                        i32_max_sorted,
                        scale_per_row=model_dim // 32,
                        scaleN_pad=((model_dim // 32 + 7) // 8) * 8,
                    )
                    emit_phase_barrier(
                        arg_desc,
                        fx.Int32(_BSTEP) * epoch + fx.Int32(_BSTEP - 1),
                        bx_i32,
                        grid_nb,
                        tx_i32,
                        tp_size=tp_size,
                    )
                g1_bound = g1_total_m * fx.Int32(g1_n_blocks)
                for raw in range(bx_i32, g1_bound, grid_nb):
                    gpu.barrier()
                    TILE = _g1_flat_tile(fx.Int32(raw), g1_bound)
                    emit_gemm1_tile(
                        arg_aq,
                        arg_ascale,
                        arg_w1,
                        arg_w1_scale,
                        arg_eids,
                        arg_mind,
                        arg_aqout,
                        arg_ascaleout,
                        arg_hidden,
                        arg_bias1,
                        TILE,
                        lane,
                        wave,
                        i32_ntok,
                        g1_total_m,
                        lds_raw,
                    )
                emit_phase_barrier(
                    arg_desc,
                    fx.Int32(_BSTEP) * epoch + fx.Int32(_BSTEP),
                    bx_i32,
                    grid_nb,
                    tx_i32,
                    tp_size=tp_size,
                )
                g2_bound = g1_total_m * fx.Int32(G2_N_BLOCKS)
                for raw2 in range(bx_i32, g2_bound, grid_nb):
                    gpu.barrier()
                    unit = fx.Int32(raw2)
                    MBLK = _udiv(unit, fx.Int32(G2_N_BLOCKS))
                    NBLK = unit - MBLK * fx.Int32(G2_N_BLOCKS)
                    emit_gemm2_tile(
                        arg_aqout,
                        arg_ascaleout,
                        arg_w2,
                        arg_w2_scale,
                        arg_eids,
                        arg_stids,
                        arg_sweights,
                        arg_bias2,
                        arg_out,
                        MBLK,
                        NBLK,
                        lane,
                        wave,
                        i32_M,
                        i32_max_m_blocks,
                        i32_inter,
                        i32_hidden,
                        lds,
                    )
            else:

                def _do_shuffle(mb):
                    if fuse_ag and _ASCALE_SHUFFLE:
                        emit_ascale_shuffle(
                            arg_ascale_raw,
                            arg_ascale,
                            arg_stids,
                            arg_cumsum,
                            i32_ntok,
                            mb * fx.Int32(g1_BM),
                            fx.Int32(1),
                            tx_i32,
                            fx.Int32(_BLOCK),
                            (mb + fx.Int32(1)) * fx.Int32(g1_BM),
                            scale_per_row=model_dim // 32,
                            scaleN_pad=((model_dim // 32 + 7) // 8) * 8,
                        )
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        gpu.barrier()

                def _g1_flat_one(raw):
                    """One GEMM1 tile addressed by its flattened (m, n) index."""
                    gpu.barrier()
                    emit_gemm1_tile(
                        arg_aq, arg_ascale, arg_w1, arg_w1_scale, arg_eids,
                        arg_mind, arg_aqout, arg_ascaleout, arg_hidden,
                        arg_bias1,
                        _g1_flat_tile(raw, g1_total_m * fx.Int32(g1_n_blocks)),
                        lane, wave, i32_ntok, g1_total_m, lds_raw,
                    )

                def _do_gemm1(mb):
                    for nb1 in range_constexpr(g1_n_blocks):
                        gpu.barrier()
                        emit_gemm1_tile(
                            arg_aq,
                            arg_ascale,
                            arg_w1,
                            arg_w1_scale,
                            arg_eids,
                            arg_mind,
                            arg_aqout,
                            arg_ascaleout,
                            arg_hidden,
                            arg_bias1,
                            mb * fx.Int32(g1_n_blocks)
                            + (nb1 if _ROLL else fx.Int32(nb1)),
                            lane,
                            wave,
                            i32_ntok,
                            g1_total_m,
                            lds_raw,
                        )

                def _g2_tile_rt(mb, nb2):
                    """One GEMM2 tile; consecutive tiles share one LDS region."""
                    gpu.barrier()
                    emit_gemm2_tile(
                        arg_aqout, arg_ascaleout, arg_w2, arg_w2_scale,
                        arg_eids, arg_stids, arg_sweights, arg_bias2,
                        arg_out,
                        mb, nb2, lane, wave, i32_M, i32_max_m_blocks,
                        i32_inter, i32_hidden, lds,
                    )

                def _g2_tile(mb, nb2, par):
                    _g2_tile_rt(mb, nb2)

                def _do_gemm2(mb):
                    n2 = G2_N_BLOCKS
                    if _ROLL and n2:
                        for nb2 in range(
                            fx.Int32(0), _udiv(i32_hidden, fx.Int32(g2_BN)),
                            fx.Int32(1),
                        ):
                            _g2_tile_rt(mb, nb2)
                    else:
                        for nb2 in range_constexpr(n2):
                            _g2_tile(mb, fx.Int32(nb2), nb2)

                for raw in range(bx_i32, g1_total_m, grid_nb):
                  mb = _m_block(fx.Int32(raw))
                  _do_shuffle(mb)
                  _do_gemm1(mb)
                  rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                  gpu.barrier()
                  _do_gemm2(mb)

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
        def launch_stage12_rs(
            arg_hidden: fx.Int64,
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1_scale: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2_scale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            arg_mind: fx.Int64,
            arg_bias1: fx.Int64,
            arg_bias2: fx.Int64,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_out: fx.Int64,
            i32_ntok: fx.Int32,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_desc: fx.Int64,
            i32_rank: fx.Int32,
            i32_rows: fx.Int32,
            arg_ag_desc: fx.Int64,
            arg_ascale_raw: fx.Int64,
            i32_max_sorted: fx.Int32,
            arg_tk_ids: fx.Int64,
            arg_tk_weights: fx.Int64,
            arg_num_valid: fx.Int64,
            arg_dbg: fx.Int64,
            arg_taskq: fx.Int64,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            stage12_kernel(
                arg_hidden,
                arg_aq,
                arg_ascale,
                arg_w1,
                arg_w1_scale,
                arg_w2,
                arg_w2_scale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_mind,
                arg_bias1,
                arg_bias2,
                arg_aqout,
                arg_ascaleout,
                arg_out,
                i32_ntok,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                arg_desc,
                i32_rank,
                i32_rows,
                arg_ag_desc,
                arg_ascale_raw,
                i32_max_sorted,
                arg_tk_ids,
                arg_tk_weights,
                arg_num_valid,
                arg_dbg,
                arg_taskq,
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1),
                block=(_BLOCK, 1, 1),
                stream=stream,
            )

        launch_stage12_rs.block = _BLOCK
        launch_stage12_rs.g2_n_blocks = G2_N_BLOCKS
        return launch_stage12_rs

    return compile_gemm2_a4w4_port(
        BM=g2_BM,
        BN=g2_BN,
        BK=g2_BK,
        use_nt=g2_use_nt,
        HIDDEN_MAX=HIDDEN_MAX,
        epilog=g2_epilog,
        INTER_MAX=INTER_MAX,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        topk=topk if g2_epilog == "reduce" else 1,
        SBM=g2_SBM,
        persist=False,
        g2_spart=g2_spart,
        g2_bf16_lds=g2_bf16_lds,
        g2_kstatic=g2_kstatic,
        out_dtype="bf16",
        enable_bias=False,
        _composition=g2_compose,
    )


class _G1Handle:
    """Stand-in for the launcher ``compile_gemm1_a4w4_port`` would return."""

    compile_hints: dict = {}


def _ptr(t, fallback):
    """Device address of ``t``, or of ``fallback`` when the kernel ignores it."""
    return int((fallback if t is None else t).data_ptr())


def _mind_ptr(m_indices, sorted_token_ids):
    """Address GEMM1 should read ``m_indices`` from."""
    return int((m_indices if m_indices is not None else sorted_token_ids).data_ptr())


def _dbg_ptr(dbg, sorted_token_ids):
    return int((dbg if dbg is not None else sorted_token_ids).data_ptr())


_TASKQ_BUFS: dict = {}


def _taskq_ptr(sorted_token_ids, max_sorted, g2_BM):
    """Device address of the task queue's counters."""
    blocks = int((int(max_sorted) + int(g2_BM) - 1) // int(g2_BM))
    key = (blocks, str(sorted_token_ids.device))
    buf = _TASKQ_BUFS.get(key)
    if buf is None:
        buf = torch.zeros(1 + 2 * blocks, dtype=torch.int32,
                          device=sorted_token_ids.device)
        _TASKQ_BUFS[key] = buf
    return int(buf.data_ptr())


def _as_u8(t):
    if t is not None and t.element_size() == 1 and t.dtype != torch.uint8:
        return t.view(torch.uint8)
    return t


def run_stage12_rs(
    *,
    g1,
    w2,
    w2_scale,
    sorted_token_ids,
    sorted_weights,
    partial,
    output,
    desc_ptr,
    rank,
    tp_size,
    local_rows,
    M_logical,
    model_dim,
    inter_dim,
    g2_cfg,
    block_m=None,
    ag_desc_ptr=0,
    ascale_raw=None,
    topk=0,
    fuse_ag=False,
    waves_per_eu=None,
    tk_ids=None,
    tk_weights=None,
    num_valid=None,
    dbg=None,
    stream=None,
):
    """Launch GEMM1 + GEMM2 + ReduceScatter as one kernel."""
    g2_BM = g2_cfg["tile_m"]
    g2_SBM = g2_cfg["sort_block_m"] or (int(block_m) if block_m else g2_BM)
    wpe = _WAVES_PER_EU if waves_per_eu is None else int(waves_per_eu)
    kstatic = os.environ.get("MXFP4_G2_KSTATIC", "1") == "1"
    launch = compile_stage12_rs(
        int(tp_size),
        int(model_dim),
        g1_BM=int(g1["BM"]),
        g1_use_nt=bool(g1["use_nt"]),
        g1_inline_quant=bool(g1["inline_quant"]),
        g1_act=g1["act"],
        g1_situ_beta=float(g1["situ_beta"]),
        g1_situ_linear_beta=float(g1["situ_linear_beta"]),
        g1_swiglu_limit=float(g1["swiglu_limit"]),
        g1_native_scale_layout=bool(g1["native_scale_layout"]),
        g1_interleave=bool(g1["interleave"]),
        g1_xcd_swizzle=int(g1["xcd_swizzle"]),
        g1_num_waves=int(g1["num_waves"]),
        g1_k_wave=int(g1["k_wave"]),
        D_HIDDEN=int(g1["D_HIDDEN"]),
        D_INTER=int(g1["D_INTER"]),
        NE=int(g1["NE"]),
        g2_BM=g2_BM,
        g2_BN=g2_cfg["tile_n"],
        g2_BK=g2_cfg["tile_k"],
        g2_use_nt=bool(g2_cfg["use_nt"]),
        g2_SBM=g2_SBM,
        g2_spart=g2_cfg["spart"],
        g2_bf16_lds=g2_cfg["bf16_lds"],
        g2_kstatic=kstatic,
        HIDDEN_MAX=8192,
        INTER_MAX=int(inter_dim) if kstatic else 8192,
        a_dtype=g2_cfg["a_dtype"],
        b_dtype=g2_cfg["b_dtype"],
        topk=int(topk),
        fuse_ag=bool(fuse_ag),
        g2_epilog=g2_cfg["epilog"],
        waves_per_eu=wpe,
    )

    aqout = g1["inter_sorted_quant"]
    max_sorted = int(sorted_token_ids.shape[0])
    grid_blocks = (max_sorted + g2_BM - 1) // g2_BM
    if fuse_ag:
        routes = int(M_logical) * max(1, int(topk))
        live = min(int(g1["NE"]), routes) + (routes + g2_BM - 1) // g2_BM
        grid_blocks = min(grid_blocks, max(1, live))
    if fuse_ag and _AG_GRID_CAP > 0:
        grid_blocks = min(grid_blocks, _AG_GRID_CAP)
    if _GRID_CU > 0:
        grid_blocks = _GRID_CU
    hints = {"waves_per_eu": wpe} if wpe else {}
    with CompilationContext.compile_hints(hints):
        run_compiled(
            launch,
            int(g1["hidden_states"].data_ptr()),
            _ptr(g1["a_quant"], g1["hidden_states"]),
            _ptr(g1["a_scale_sorted_shuffled"], g1["hidden_states"]),
            int(_as_u8(g1["w1_u8"]).data_ptr()),
            int(_as_u8(g1["w1_scale_u8"]).data_ptr()),
            int(_as_u8(w2).data_ptr()),
            int(_as_u8(w2_scale).data_ptr()),
            int(g1["sorted_expert_ids"].data_ptr()),
            int(g1["cumsum_tensor"].data_ptr()),
            int(sorted_token_ids.data_ptr()),
            int(sorted_weights.data_ptr()),
            _mind_ptr(g1["m_indices"], sorted_token_ids),
            _ptr(g1["bias"], partial),
            int(partial.data_ptr()),
            int(aqout.data_ptr()),
            int(g1["inter_sorted_shuffled_scale"].data_ptr()),
            int(partial.data_ptr()),
            int(g1["n_tokens"]),
            int(M_logical),
            int((max_sorted + g2_BM - 1) // g2_BM),
            int(inter_dim),
            int(model_dim),
            int(desc_ptr),
            int(rank),
            int(local_rows),
            int(ag_desc_ptr),
            _ptr(ascale_raw, sorted_token_ids),
            int(max_sorted),
            _ptr(tk_ids, sorted_token_ids),
            _ptr(tk_weights, sorted_token_ids),
            _ptr(num_valid, sorted_token_ids),
            _dbg_ptr(dbg, sorted_token_ids),
            _taskq_ptr(sorted_token_ids, max_sorted, g2_BM),
            int(grid_blocks),
            stream if stream is not None else torch.cuda.current_stream(),
        )
    return output[:local_rows]

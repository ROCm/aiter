# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEMM1 + GEMM2 + ReduceScatter in one kernel launch.

:mod:`.stage2_rs` folds the ReduceScatter into GEMM2. This goes one step
further and hosts *both* GEMMs, so the tail of the TP MoE layer collapses from

    gemm1 -> gemm2 -> rs_publish -> rs_pull        (4 launches)

to a single one. Neither GEMM is reimplemented: both come from the
``_composition`` hook on their own compiler
(:func:`~..mxfp4_gemm1.compile_gemm1_a4w4_port` and
:func:`~..mxmoe_dispatcher.compile_gemm2_a4w4_port`), so the tuned tiles are
emitted verbatim.

    phase 1   persistent grid-stride over GEMM1 tiles -> sorted FP4 intermediate
    -- grid-wide barrier, agent-scope release/acquire --
    phase 2   persistent grid-stride over GEMM2 tiles -> arena partial
    tail      the shared ReduceScatter tail from :mod:`.rs_tail`

Why this pair and not any pair
------------------------------
The two GEMMs must agree on a block size, because one kernel has one. GEMM2 is
always 256 threads; GEMM1 is ``num_waves * k_wave * 64``, which is 256 exactly
when ``num_waves=4, k_wave=1`` -- the BM16 inline-quant rows small ``M`` tunes
onto. :func:`stage12_supported` checks that rather than assuming it. LDS is the
union of the two (16640 B and 8192 B for kimi3 BM16), allocated once and handed
to both, since the phases are disjoint.

The grid is persistent and sized to what the device holds at once: the
barrier between phases makes that a correctness requirement, not a tuning
choice.

The handoff needs a real agent-scope release/acquire, not the bare
``s_waitcnt`` the RS tail uses -- MI355X L2 is per-XCD, so GEMM1's stores are
not visible to a GEMM2 tile on another XCD without it. See
:func:`~.rs_tail.emit_phase_barrier`.
"""

# NOTE: no ``from __future__ import annotations`` here. It would turn the
# ``@fx.struct`` field annotation below into a string, and FlyDSL resolves those
# eagerly -- the failure is "type fx.Array[...] does not implement the Storable
# protocol". The other kernel modules omit it for the same reason.

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import Int8, T

from ..mxfp4_gemm1 import compile_gemm1_a4w4_port
from ..mxfp4_gemm_common import _udiv, _umod, global_typed_ptr
from ..mxmoe_dispatcher import _spart_output_tile_index, compile_gemm2_a4w4_port
from ..tensor_shim import _run_compiled as run_compiled
from .reduce_scatter import MAX_SERVICE_BLOCKS, RS_UNIT_ELEMS
from .rs_tail import emit_phase_barrier, emit_rs_tail, read_epoch, rs_tail_slots

__all__ = ["compile_stage12_rs", "run_stage12_rs", "stage12_supported"]

_BLOCK = 256
_SERVICE_BLOCKS = min(
    MAX_SERVICE_BLOCKS, int(os.environ.get("AITER_TP_STAGE12_RS_SERVICE", "128"))
)
#: CTAs to launch. The inter-phase barrier requires every CTA resident, so this
#: is the device's CU count rather than the tile count.
_GRID_CU = int(os.environ.get("AITER_TP_STAGE12_GRID", "0"))


def _grid_ctas() -> int:
    if _GRID_CU > 0:
        return _GRID_CU
    from aiter.jit.utils.chip_info import get_cu_num

    return int(get_cu_num())


def stage12_supported(g1_cfg, g2_cfg, model_dim: int) -> bool:
    """Whether this tuned (GEMM1, GEMM2) pair can share one kernel.

    The binding constraint is the block size: one kernel has one, GEMM2 is
    always 256 threads, and GEMM1 is ``num_waves * k_wave * 64``.
    """
    if g1_cfg is None or g2_cfg is None:
        return False
    if model_dim % RS_UNIT_ELEMS:
        return False
    if g1_cfg.get("a_dtype") != "fp4" or g1_cfg.get("out_dtype") != "fp4":
        return False
    if int(g1_cfg.get("num_waves", 4)) * int(g1_cfg.get("k_wave", 1)) * 64 != _BLOCK:
        return False
    if g2_cfg.get("epilog") != "atomic" or g2_cfg.get("persist"):
        return False
    return True


@functools.cache
def compile_stage12_rs(
    tp_size: int,
    model_dim: int,
    *,
    # -- GEMM1 -----------------------------------------------------------
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
    # -- GEMM2 -----------------------------------------------------------
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
    service_blocks: int = _SERVICE_BLOCKS,
):
    """Build the fused GEMM1 + GEMM2 + ReduceScatter launcher for one row pair."""
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")

    # Mirror the GEMM2 knob resolution so the partitioner replayed here matches
    # the one the tile was compiled with.
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0
    tail_slots = rs_tail_slots(tp_size)
    grid_ctas = _grid_ctas()

    # -- collect the GEMM1 tile emitter ----------------------------------
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

    # -- build the merged kernel inside the GEMM2 composition -------------
    def g2_compose(*, module_name, emit_gemm2_tile, shared_storage, lds_bytes, **_):
        merged_lds = max(int(lds_bytes), g1_lds_bytes)
        name = (
            f"mega_moe_tp_stage12_rs_tp{tp_size}_h{model_dim}"
            f"_g1bm{g1_BM}_g2bm{g2_BM}x{g2_BN}_sv{service_blocks}"
        )

        @fx.struct
        class MergedStorage:
            # One region for both phases: they never overlap in time, so the
            # union is enough and the two bodies each take the raw base.
            buf: fx.Array[Int8, merged_lds, 16]

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
        ):
            tx_i32 = fx.Int32(gpu.thread_id("x"))
            bx_i32 = fx.Int32(gpu.block_id("x"))
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            grid_nb = fx.Int32(gpu.grid_dim.x)
            lds = fx.SharedAllocator().allocate(MergedStorage).peek()
            lds_raw = lds.buf.ptr

            epoch_addr, epoch = read_epoch(arg_desc, tail_slots)
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]

            # -- phase 1: GEMM1 ---------------------------------------------
            g1_total_m = _udiv(cumsum0, g1_BM)
            g1_bound = g1_total_m * fx.Int32(g1_n_blocks)
            _NXCD = 8
            _xq = _udiv(g1_bound, _NXCD)
            _xr = _umod(g1_bound, _NXCD)

            def _g1_tile(pid):
                if const_expr(g1_xcd_swizzle <= 0):
                    return pid
                xc = _umod(pid, _NXCD)
                wgid = xc * _xq + fx.min(xc, _xr) + _udiv(pid, _NXCD)
                ng = fx.Int32(g1_xcd_swizzle * g1_n_blocks)
                group_id = wgid // ng
                first_pid_m = group_id * fx.Int32(g1_xcd_swizzle)
                remaining_m = g1_total_m - first_pid_m
                group_size_m = fx.min(remaining_m, fx.Int32(g1_xcd_swizzle))
                wig = wgid % ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(g1_n_blocks) + n_block

            for raw in range(bx_i32, g1_bound, grid_nb):
                # Unconditional: this is a runtime loop, so a Python-level
                # "skip the first iteration" flag would be evaluated once at
                # trace time and emit nothing. The extra barrier on iteration
                # zero is harmless; reusing LDS across iterations is not.
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
                    _g1_tile(fx.Int32(raw)),
                    lane,
                    wave,
                    i32_ntok,
                    g1_total_m,
                    lds_raw,
                )

            # -- handoff -----------------------------------------------------
            emit_phase_barrier(
                arg_desc, epoch, bx_i32, grid_nb, tx_i32, tp_size=tp_size
            )

            # -- phase 2: GEMM2 ----------------------------------------------
            num_n_blocks = fx.Int32(fx.Uint32(i32_hidden) // fx.Uint32(g2_BN))
            g2_total_m = _udiv(cumsum0, g2_BM)
            g2_bound = g2_total_m * num_n_blocks
            for raw2 in range(bx_i32, g2_bound, grid_nb):
                gpu.barrier()
                unit = fx.Int32(raw2)
                if const_expr(g2_spart > 0):
                    m_block_idx, n_block_idx = _spart_output_tile_index(
                        unit, g2_total_m, num_n_blocks, g2_group_num, g2_m01
                    )
                else:
                    m_block_idx = _udiv(unit, num_n_blocks)
                    n_block_idx = unit - m_block_idx * num_n_blocks
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

            # -- ReduceScatter -----------------------------------------------
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
            ).launch(
                grid=(fx.Int64(grid_ctas), 1, 1),
                block=(_BLOCK, 1, 1),
                stream=stream,
            )

        launch_stage12_rs.block = _BLOCK
        launch_stage12_rs.grid_ctas = grid_ctas
        return launch_stage12_rs

    return compile_gemm2_a4w4_port(
        BM=g2_BM,
        BN=g2_BN,
        BK=g2_BK,
        use_nt=g2_use_nt,
        HIDDEN_MAX=HIDDEN_MAX,
        epilog="atomic",
        INTER_MAX=INTER_MAX,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        topk=1,
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
    """Stand-in for the launcher ``compile_gemm1_a4w4_port`` would return.

    The GEMM1 compiler sets ``compile_hints`` on whatever a composition hands
    back; this kernel is launched through the GEMM2 side, so that object is
    discarded and only needs to accept the attribute.
    """

    compile_hints: dict = {}


def _ptr(t, fallback):
    """Device address of ``t``, or of ``fallback`` when the kernel ignores it."""
    return int((fallback if t is None else t).data_ptr())


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
    stream=None,
):
    """Launch GEMM1 + GEMM2 + ReduceScatter as one kernel.

    ``g1`` is the keyword dict ``_mxfp4_a4w4_stage1`` would have handed to
    ``flydsl_mxfp4_gemm1`` -- captured through its ``_gemm1_launch`` hook, so
    every operand and every compile knob is the one the tuned path derived
    rather than a copy of that derivation.
    """
    g2_BM = g2_cfg["tile_m"]
    g2_SBM = g2_cfg["sort_block_m"] or (int(block_m) if block_m else g2_BM)
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
    )

    aqout = g1["inter_sorted_quant"]
    max_sorted = int(sorted_token_ids.shape[0])
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
        _ptr(g1["m_indices"], sorted_token_ids),
        _ptr(g1["bias"], partial),
        int(partial.data_ptr()),  # unused bias2; any mapped address
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
        stream if stream is not None else torch.cuda.current_stream(),
    )
    return output[:local_rows]

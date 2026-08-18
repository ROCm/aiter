# SPDX-License-Identifier: MIT
"""H1 prototype: resident copy/put plus A4W4 GMM1+activation+A4."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops
from ..gemm1 import (
    MXFP4_SCALE_LAYOUT_TAG,
    _bm_constants,
    _gemm1_body,
    _global_i32_at,
    k_tiles_total_for,
    n_out_for,
    num_n_blocks_for,
)

from ..activation import normalize_activation, validate_activation_parameters


def compile_hier_stage1_a4w4(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    COMM_BLOCKS: int = 1,
    BM: int = 32,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = True,
    enable_copy: bool = True,
    enable_signal: bool = True,
    rejoin_compute: bool = True,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    """Compile the first fused H1 skeleton.

    Low block IDs copy one already-packed chunk and publish a system-release
    generation. Remaining blocks run the activation-specialized A4W4 GMM1
    body. The activation selection is compile-time and does not add a runtime
    branch around MFMA. The next revision replaces the raw copy role with MORI
    put+signal and adds the destination tile-ready queue; the compute body and
    shared-resource union do not change.
    """

    activation = normalize_activation(activation)
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if BN != 256 or BK != 256:
        raise ValueError(f"H1 A4W4 body requires BN=BK=256, got BN={BN}, BK={BK}")
    if (BM, use_nt) not in {
        (32, True),
        (32, False),
        (64, False),
        (128, False),
    }:
        raise ValueError(
            f"unsupported H1 A4W4 variant BM={BM}, use_nt={use_nt}"
        )
    if D_HIDDEN % BK or (2 * D_INTER) % BN:
        raise ValueError("H1 dimensions must be divisible by BK/BN")
    if COMM_BLOCKS < 1:
        raise ValueError("COMM_BLOCKS must be positive")

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(D_HIDDEN, BK)
    _, _, _, lds_bytes = _bm_constants(BM, BN, kh_tile, k_tiles_total)
    num_n_blocks = num_n_blocks_for(n_out_for(D_INTER), BN)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    def float_tag(value: float) -> str:
        return (
            str(float(value))
            .replace("-", "m")
            .replace("+", "p")
            .replace(".", "p")
        )

    activation_tag = activation
    if activation == "swiglu":
        limit = 7.0 if swiglu_limit is None else float(swiglu_limit)
        activation_tag += f"_l{float_tag(limit)}"
    elif activation == "situv2":
        beta_tag = float_tag(situ_beta)
        linear_tag = float_tag(situ_linear_beta)
        activation_tag += f"_b{beta_tag}_lb{linear_tag}"
    elif swiglu_limit is not None:
        limit_tag = float_tag(swiglu_limit)
        activation_tag += f"_l{limit_tag}"

    name = (
        f"megamoe_tile_h1_a4w4_{activation_tag}_h{D_HIDDEN}_i{D_INTER}_"
        f"e{NE}_k{TOPK}_bm{BM}_cb{COMM_BLOCKS}_"
        f"cp{int(enable_copy)}_sig{int(enable_signal)}_rj{int(rejoin_compute)}_"
        f"{MXFP4_SCALE_LAYOUT_TAG}"
    )
    # Keep the proven default symbol stable; only the alternate cache policy
    # needs a tag to avoid aliasing the BM32 use_nt=True code object.
    if not use_nt:
        name += "_nt0"

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def h1_kernel(
        # communication stub
        copy_src: fx.Int64,
        copy_dst: fx.Int64,
        copy_signal: fx.Int64,
        copy_nbytes: fx.Int32,
        generation: fx.Int64,
        # existing sorted A4W4 GMM1 ABI
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))

        is_comm = bx < fx.Int32(COMM_BLOCKS)
        if is_comm:
            if const_expr(enable_copy):
                src = buffer_ops.create_buffer_resource_from_addr(copy_src)
                dst = buffer_ops.create_buffer_resource_from_addr(copy_dst)
                units = fx.Int32(copy_nbytes) // fx.Int32(16)
                for unit in range(tx, units, fx.Int32(256)):
                    elem = unit * fx.Int32(4)
                    value = buffer_ops.buffer_load(src, elem, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(value, dst, elem)
                tail = units * fx.Int32(16) + tx
                if tail < fx.Int32(copy_nbytes):
                    value = buffer_ops.buffer_load(src, tail, vec_width=1, dtype=T.i8)
                    buffer_ops.buffer_store(value, dst, tail)
            if const_expr(enable_signal):
                gpu.barrier()
                if tx == fx.Int32(0):
                    comm_ops.store_i64_global_system(copy_signal, generation)

        def run_gemm(tile):
            # The production path will wait on this tile's ready generation and
            # issue one matching system acquire there. Do not place an
            # unconditional agent fence on every GMM tile: it serializes the
            # whole grid and destroys communication/compute overlap.
            total_m_blocks = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
            _gemm1_body(
                lds_raw_ptr,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                tile,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=False,
                K=D_HIDDEN,
                N_OUT=2 * D_INTER,
                NE=NE,
                interleave=False,
                act=activation,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )

        if const_expr(rejoin_compute):
            # Communication roles rejoin the common compute path. Keeping the
            # MFMA body outside the role branch is essential for LLVM/AMDGPU
            # scheduling quality (the nested-else form is ~6x slower).
            run_gemm(bx)
        else:
            if bx >= fx.Int32(COMM_BLOCKS):
                run_gemm(bx - fx.Int32(COMM_BLOCKS))

    @flyc.jit
    def launch_h1_sc2(
        copy_src: fx.Int64,
        copy_dst: fx.Int64,
        copy_signal: fx.Int64,
        copy_nbytes: fx.Int32,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_gemm_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        h1_kernel(
            copy_src,
            copy_dst,
            copy_signal,
            copy_nbytes,
            generation,
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(
            grid=(
                i32_gemm_grid
                if const_expr(rejoin_compute)
                else fx.Int32(COMM_BLOCKS) + i32_gemm_grid,
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_h1_sc2.kernel_name = name
    launch_h1_sc2.lds_bytes = lds_bytes
    launch_h1_sc2.num_n_blocks = num_n_blocks
    return launch_h1_sc2


def compile_hier_stage1_a4w4_silu(**kwargs):
    """Backward-compatible wrapper for the original SiLU specialization."""

    requested = normalize_activation(kwargs.pop("activation", "silu"))
    if requested != "silu":
        raise ValueError(
            "compile_hier_stage1_a4w4_silu only accepts activation='silu'; "
            "use compile_hier_stage1_a4w4 for other activations"
        )
    return compile_hier_stage1_a4w4(activation="silu", **kwargs)

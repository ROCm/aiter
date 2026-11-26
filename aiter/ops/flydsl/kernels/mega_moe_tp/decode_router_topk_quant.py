# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Top-4 plus MXFP4 / MXFP8 input quant in one launch (<= 256 rows).

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_topk_mxfp4_mxfp8_quant_sorted_tasks.py`` (Apache-2.0, "Copyright contributors to the
amd-kernels project"); the kernel bodies are unchanged, only imports, the warm-up ``@launcher`` wrapper and the
compile-cache helper were replaced by aiter's own (``aiter.ops.flydsl.kernels.buffer_ops`` / ``tensor_shim``).
"""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .decode_router_common import (
    DPP_TREE,
    MAX_BUFFER_BYTES,
    NEG_LOG2E,
    NW,
    PRELOAD_HINTS,
    _as_f32,
    _cvt_pk_fp4,
    _dpp,
    _e8m0_even_e2m1,
    _fmax3_abs,
    _ir,
    on_supported_device,
    run_on_device,
)
from .decode_tile_primitives import _e8m0_fp8, _fp8

H = 6144
E, TOPK = 128, 4
BLOCKS = H // 32  # 32-value blocks per row
assert BLOCKS % 64 == 0 and NW == 1


# DPP steps whose source lane is valid in every lane (quad_perm, row_half_mirror, row_mirror)
FULL_DPP = (0xB1, 0x4E, 0x141, 0x140)


def _dpp_full(v, ctrl):
    """``_dpp`` for a step with a valid source in every lane: with bound_ctrl the old value is
    not read, so LLVM needs no copy of ``v`` before the v_mov_b32_dpp (same result)."""
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, _ir(v), _ir(v), ctrl, 0xF, 0xF, True))


def _maxf_nn(a, b):
    """v_max_f32 with nnan (the queue holds no NaN): no canonicalization, and the value does not wait for the index."""
    fm = _mlir_ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def compile_router_topk_mxfp4_mxfp8_quant_sorted_tasks(
    num_experts: int,
    topk: int,
    split_k: int,
    token_major: bool = False,
    quant_only: bool = False,
):
    """Each lane sorts its four choices once; the winning lane pops its head after each pick.

    Strict adjacent swaps keep the lower component first on ties. The tree's sixth DPP step only copies lane 31's
    result to lane 63, so the tree stops at five steps and reads lane 31.

    Args:
        token_major: contiguous [m, S, 128] partials, read at constant split offsets.
        quant_only: only the quant workgroups (no top-k, no partials / bias / outputs read or written): the input
            quant of a caller whose top-k comes from elsewhere (``quant_mxfp4_mxfp8``).
    """
    assert num_experts == 128 and topk == 4
    E, K, S = num_experts, topk, split_k
    assert S % 8 == 0 and 8 <= S <= 96
    name = (
        f"router_splitk_biased_top{K}_mxfp4_mxfp8_quant_sorted_tasks_h{H}_e{E}_s{S}"
        + "_half_tree_maxnum_direct_weight"
        + ("_token_partials" if token_major else "")
        + ("_quant_only" if quant_only else "")
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_part: fx.Int64,  # split-K partials, fp32, [M, S, E] by strides
        arg_bias: fx.Int64,  # correction bias [E] fp32
        arg_w: fx.Int64,  # top-k weights fp32, row stride i32_stk
        arg_ids: fx.Int64,  # top-k ids int32, row stride i32_stk
        i64_part_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sm: fx.Int32,  # partial strides (elements): token, split
        i32_sk: fx.Int32,
        i32_stk: fx.Int32,
        i32_scale: fx.Int32,  # routed_scaling_factor (fp32 bits)
        arg_x: fx.Int64,  # bf16 [M, H] row-major (the router GEMM's input)
        arg_q: fx.Int64,  # e2m1 codes [M, H / 2]
        arg_s: fx.Int64,  # routed e8m0 [M,H/32]
        arg_q8: fx.Int64,
        arg_s8: fx.Int64,
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        t = pid * NW + wave
        top_m = fx.Int32(0) if const_expr(quant_only) else i32_m
        if const_expr(not quant_only):  # noqa: SIM102 -- DSL trace-time region
            if t < i32_m:
                pr = buffer_ops.create_buffer_resource_from_addr(
                    arg_part, num_records_bytes=i64_part_bytes
                )
                br = buffer_ops.create_buffer_resource_from_addr(
                    arg_bias, num_records_bytes=E * 4
                )
                out_bytes = fx.Int64(i32_m) * fx.Int64(i32_stk) * 4
                wr = buffer_ops.create_buffer_resource_from_addr(
                    arg_w, num_records_bytes=out_bytes
                )
                ir = buffer_ops.create_buffer_resource_from_addr(
                    arg_ids, num_records_bytes=out_bytes
                )
                has = lane < E // 4
                el = lane % 32 * 4 + lane // 32 * 2
                base = t * (S * E if const_expr(token_major) else i32_sm) + el
                parts = [
                    Vec(
                        buffer_ops.buffer_load(
                            pr,
                            base + k * (E if const_expr(token_major) else i32_sk),
                            vec_width=2,
                            dtype=fx.Float32,
                        )
                    )
                    for k in range_constexpr(S)
                ]
                bias = Vec(
                    buffer_ops.buffer_load(
                        br, lane * 4, vec_width=4, dtype=fx.Float32, mask=has
                    )
                )
                neg_inf = fx.Float32(float("-inf"))
                ss = []
                for i in range_constexpr(2):
                    x = fx.Float32(0.0)
                    for k in range_constexpr(S):
                        x = x + fx.Float32(parts[k][i])
                    y = (x.to(fx.Float64) * fx.Float64(NEG_LOG2E)).to(fx.Float32)
                    ss.append(
                        fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(1.0) + y.exp2()))
                    )
                # components 2 / 3 back to the quartet's lower lane, for the same argmax tree and ties
                sig = [
                    ss[0],
                    ss[1],
                    fx.Int32(
                        fx.rocdl.ds_bpermute(
                            T.i32, (lane ^ 32) * 4, ss[0].bitcast(fx.Int32)
                        )
                    ).bitcast(fx.Float32),
                    fx.Int32(
                        fx.rocdl.ds_bpermute(
                            T.i32, (lane ^ 32) * 4, ss[1].bitcast(fx.Int32)
                        )
                    ).bitcast(fx.Float32),
                ]
                choice = [
                    has.select(sig[i] + fx.Float32(bias[i]), neg_inf)
                    for i in range_constexpr(4)
                ]
                # Strict argmax ignores NaNs and -inf; keep that behavior in the queue.
                choice = [(v > neg_inf).select(v, neg_inf) for v in choice]
                local_ids = [lane * 4 + i for i in range_constexpr(4)]
                # stable insertion network: ties keep the component order
                for a, b in ((0, 1), (1, 2), (0, 1), (2, 3), (1, 2), (0, 1)):
                    take = choice[b] > choice[a]
                    va, vb = choice[a], choice[b]
                    ia, ib = local_ids[a], local_ids[b]
                    choice[a], choice[b] = take.select(vb, va), take.select(va, vb)
                    local_ids[a], local_ids[b] = take.select(ib, ia), take.select(
                        ia, ib
                    )
                total = fx.Float32(0.0)
                out_id = fx.Int32(0)
                out_val = fx.Float32(0.0)
                for k in range_constexpr(K):
                    mv = choice[0]
                    mi = (mv > neg_inf).select(local_ids[0], fx.Int32(k))
                    for ctrl in DPP_TREE[:-1]:
                        dpp = _dpp_full if ctrl in FULL_DPP else _dpp
                        rv = dpp(mv.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                        ri = dpp(mi, ctrl)
                        take = rv > mv
                        mv = _maxf_nn(rv, mv)
                        mi = take.select(ri, mi)
                    idx = fx.Int32(fx.rocdl.readlane(T.i32, mi, fx.Int32(31)))
                    owner, comp = idx >> 2, idx & 3
                    # expert idx: element comp % 2 of lane owner + 32 (comp // 2), also the strict argmax's fallback index
                    source_lane = owner + ((comp & 2) << 4)
                    local_w = ((comp & 1) == 1).select(ss[1], ss[0])
                    w = fx.Int32(
                        fx.rocdl.readlane(T.i32, local_w.bitcast(fx.Int32), source_lane)
                    ).bitcast(fx.Float32)
                    # Only the lane that supplied the global pick advances its queue.
                    pick = lane == owner
                    for i in range_constexpr(3):
                        choice[i] = pick.select(choice[i + 1], choice[i])
                        local_ids[i] = pick.select(local_ids[i + 1], local_ids[i])
                    choice[3] = pick.select(neg_inf, choice[3])
                    total = total + w
                    out_id = (lane == k).select(idx, out_id)
                    out_val = (lane == k).select(w, out_val)
                scale = i32_scale.bitcast(fx.Float32)
                factor = scale / total
                off = t * i32_stk + lane
                buffer_ops.buffer_store(out_val * factor, wr, off, mask=lane < K)
                buffer_ops.buffer_store(out_id, ir, off, mask=lane < K)
        if pid >= top_m:
            blk_base = (pid - top_m) * 64
            if blk_base < i32_m * BLOCKS:
                # one 32-value group per lane
                row_bytes = fx.Int64(i32_m) * (H * 2)
                xr = buffer_ops.create_buffer_resource_from_addr(
                    arg_x, num_records_bytes=row_bytes
                )
                xv = [
                    [
                        Vec(
                            buffer_ops.buffer_load(
                                xr,
                                (blk_base + lane + 64 * b) * 16 + j * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for j in range_constexpr(4)
                    ]
                    for b in range_constexpr(1)
                ]
                # both quants, quant.py's and shared_quant's arithmetic op for op
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_m) * (H // 2)
                )
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_s, num_records_bytes=fx.Int64(i32_m) * (H // 32)
                )
                q8r = buffer_ops.create_buffer_resource_from_addr(
                    arg_q8, num_records_bytes=fx.Int64(i32_m) * H
                )
                s8r = buffer_ops.create_buffer_resource_from_addr(
                    arg_s8, num_records_bytes=fx.Int64(i32_m) * (H // 32)
                )
                for b in range_constexpr(1):
                    blk = blk_base + lane + 64 * b
                    vals = []
                    for j in range_constexpr(4):
                        for kk in range_constexpr(4):
                            w = fx.Int32(xv[b][j][kk])
                            vals.append(_as_f32(w << 16))
                            vals.append(_as_f32(w & fx.Int32(-65536)))
                    amax = _fmax3_abs(vals[0], vals[1], vals[2])
                    for i in range_constexpr(3, 31, 2):
                        amax = _fmax3_abs(vals[i], vals[i + 1], amax)
                    amax = _fmax3_abs(vals[31], vals[31], amax)
                    e8 = _e8m0_fp8(amax)
                    sf8 = (e8 << 23).bitcast(fx.Float32)
                    for half in range_constexpr(2):
                        words = []
                        for dw in range_constexpr(4):
                            j = half * 16 + dw * 4
                            pk = _fp8(fx.Int32(0), vals[j], vals[j + 1], sf8, False)
                            pk = _fp8(pk, vals[j + 2], vals[j + 3], sf8, True)
                            words.append(pk)
                        buffer_ops.buffer_store(
                            Vec.from_elements(words, fx.Int32), q8r, blk * 8 + half * 4
                        )
                    buffer_ops.buffer_store(e8.to(fx.Int8), s8r, blk)
                    e8m0 = _e8m0_even_e2m1(amax)
                    scale_f = _as_f32(e8m0 << 23)
                    qo = []
                    for d in range_constexpr(4):
                        p = fx.Int32(0)
                        for kk in range_constexpr(4):
                            p = _cvt_pk_fp4(
                                p,
                                vals[8 * d + 2 * kk],
                                vals[8 * d + 2 * kk + 1],
                                scale_f,
                                kk,
                            )
                        qo.append(p)
                    buffer_ops.buffer_store(
                        Vec.from_elements(qo, fx.Int32), qr, blk * 4
                    )
                    buffer_ops.buffer_store(fx.Int32(e8m0).to(fx.Int8), sr, blk)

    @flyc.jit
    def launch_router_topk_mxfp4_mxfp8_quant_sorted_tasks(
        arg_part: fx.Int64,
        arg_bias: fx.Int64,
        arg_w: fx.Int64,
        arg_ids: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sm: fx.Int32,
        i32_sk: fx.Int32,
        i32_stk: fx.Int32,
        i32_scale: fx.Int32,
        arg_x: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_q8: fx.Int64,
        arg_s8: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_part,
            arg_bias,
            arg_w,
            arg_ids,
            i64_part_bytes,
            i32_m,
            i32_sm,
            i32_sk,
            i32_stk,
            i32_scale,
            arg_x,
            arg_q,
            arg_s,
            arg_q8,
            arg_s8,
        ).launch(
            grid=(
                fx.Int64(
                    (0 if const_expr(quant_only) else i32_m)
                    + (i32_m * BLOCKS + 63) // 64
                ),
                1,
                1,
            ),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_router_topk_mxfp4_mxfp8_quant_sorted_tasks.kernel_name = name
    launch_router_topk_mxfp4_mxfp8_quant_sorted_tasks.compile_hints = PRELOAD_HINTS
    return launch_router_topk_mxfp4_mxfp8_quant_sorted_tasks


@functools.cache
def _get_kernel(
    num_experts: int,
    topk: int,
    split_k: int,
    token_major: bool = False,
    quant_only: bool = False,
):
    return compile_router_topk_mxfp4_mxfp8_quant_sorted_tasks(
        num_experts, topk, split_k, token_major, quant_only
    )


def quant_mxfp4_mxfp8(x: torch.Tensor):
    """MXFP4 and MXFP8 input quant of bf16 [m, 6144] rows (1-256, contiguous) in one launch, the arithmetic of
    ``splitk_biased_topk_mxfp4_mxfp8_quant`` without its top-k: ((FP4 codes, e8m0), (FP8 codes, e8m0)).
    """
    m = x.shape[0]
    if not (
        x.dim() == 2
        and 0 < m <= 256
        and x.shape[1] == H
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and on_supported_device(x)
    ):
        raise ValueError(
            f"quant_mxfp4_mxfp8: needs contiguous bf16 [1-256, {H}] rows on a gfx950 device, got "
            f"{tuple(x.shape)} {x.dtype}"
        )
    dev = x.device
    q = torch.empty((m, H // 2), dtype=torch.uint8, device=dev)
    s = torch.empty((m, H // 32), dtype=torch.uint8, device=dev)
    q8 = torch.empty((m, H), dtype=torch.uint8, device=dev)
    s8 = torch.empty((m, H // 32), dtype=torch.uint8, device=dev)
    z = 0  # the top-k operands are unused: no top-k workgroup runs
    run_on_device(
        _get_kernel(E, TOPK, 8, False, True),
        dev,
        z,
        z,
        z,
        z,
        0,
        m,
        0,
        0,
        0,
        0,
        x.data_ptr(),
        q.data_ptr(),
        s.data_ptr(),
        q8.data_ptr(),
        s8.data_ptr(),
    )
    return (q, s), (q8, s8)


def splitk_biased_topk_mxfp4_mxfp8_quant(
    partials: torch.Tensor,  # [M, split_k, E] fp32 (any strides, last dim dense)
    correction_bias: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int32, same row stride
    routed_scaling_factor: float,
    x: torch.Tensor,  # bf16 [M, H] contiguous
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Top-k and both input quants from one bf16 read, 1-256 rows ->
    ((FP4 codes, e8m0 scales), (FP8 codes, e8m0 scales)). Raises ValueError for inputs outside that contract.
    """
    m, split_k, num_experts = partials.shape
    topk = topk_ids.shape[1]
    if not (
        0 < m <= 256
        and num_experts == E
        and topk == TOPK
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape == (m, H)
        and partials.dtype == torch.float32
        and partials.stride(2) == 1
        and correction_bias.dtype == torch.float32
        and correction_bias.shape == (E,)
        and correction_bias.is_contiguous()
        and topk_weights.dtype == torch.float32
        and topk_ids.dtype == torch.int32
        and topk_weights.shape == (m, TOPK)
        and topk_ids.shape == (m, TOPK)
        and topk_weights.stride(1) == 1
        and topk_ids.stride(1) == 1
        and topk_weights.stride(0) == topk_ids.stride(0) >= TOPK
        # the kernels address the output rows and the partials through 32-bit buffer descriptors, and a span of
        # 2**31 - 1 bytes is the masked-store sentinel: bound both
        and m * topk_ids.stride(0) * 4 <= MAX_BUFFER_BYTES
        and (
            ((m - 1) * partials.stride(0) + (split_k - 1) * partials.stride(1) + E) * 4
            <= MAX_BUFFER_BYTES
        )
        and on_supported_device(x, partials, correction_bias, topk_weights, topk_ids)
    ):
        raise ValueError(
            "splitk_biased_topk_mxfp4_mxfp8_quant: unsupported inputs (1-256 rows of bf16 [m, 6144] x, fp32 "
            f"[m, split_k, 128] partials, fp32 [128] bias, fp32 / int32 [m, 4] outputs on one gfx950 device): "
            f"x {tuple(x.shape)} {x.dtype}, partials {tuple(partials.shape)} {partials.dtype}"
        )
    q = torch.empty((m, H // 2), dtype=torch.uint8, device=x.device)
    s = torch.empty((m, H // 32), dtype=torch.uint8, device=x.device)
    q8 = torch.empty((m, H), dtype=torch.uint8, device=x.device)
    s8 = torch.empty((m, H // 32), dtype=torch.uint8, device=x.device)
    end = (
        (m - 1) * partials.stride(0) + (split_k - 1) * partials.stride(1) + num_experts
    )
    token_major = (
        partials.stride(0) == split_k * num_experts
        and partials.stride(1) == num_experts
    )
    launch = (
        _get_kernel(num_experts, topk_ids.shape[1], split_k, True)
        if token_major
        else _get_kernel(num_experts, topk_ids.shape[1], split_k)
    )
    run_on_device(
        launch,
        x.device,
        partials.data_ptr(),
        correction_bias.data_ptr(),
        topk_weights.data_ptr(),
        topk_ids.data_ptr(),
        end * 4,
        m,
        partials.stride(0),
        partials.stride(1),
        topk_ids.stride(0),
        struct.unpack("<i", struct.pack("<f", routed_scaling_factor))[0],
        x.data_ptr(),
        q.data_ptr(),
        s.data_ptr(),
        q8.data_ptr(),
        s8.data_ptr(),
    )
    return (q, s), (q8, s8)

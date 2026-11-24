# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels gemm/mxfp8_common/mxfp8_gemm_decode.py (the decode MXFP8 GEMM, its split-K partial-slab
# mode and the tile / lookahead choices measured for the MiniMax-M3.1 TP4 QKV projection N 2560 x K 6144).
"""gfx950 MXFP8 x MXFP8 split-K GEMM for 65-128 decode rows, no LDS staging of A or W, fp32 partial slabs out.

    slab[z, m, n] = sum_{k in third z of K} A[m, k] * W[n, k]    (E4M3 codes, per-32 E8M0 scales)

Both operands are read in the (16, 16) preshuffled layout (``shuffle_weight(w, layout=(16, 16))`` for W, the
``mxfp8_quantize_shuffled`` codes for A, 16-row instead of 16-column tiles), so one ``buffer_load`` brings a contiguous
1 KB block of either operand straight into the 64 lanes of a wave: lane (klane, r) gets K bytes ``[16*klane, +16)`` of
row / column r of a 64-K block, and two blocks (K +0 and +64) make one 8-VGPR operand of
``v_mfma_scale_f32_16x16x128_f8f6f4``. A workgroup is one wave over ``RT`` 16-row tiles and ``NI`` 16-column tiles of one
K third; W (non-temporal loads) and A blocks are issued ahead of the MFMAs as far as the registers allow. The fp32 slabs
``[KS, M, N]`` are consumed by ``qkv_reduce_insert`` (the three slabs are summed in order, rounded to bf16, then normed,
roped and inserted), so no reduce launch and no bf16 intermediate exist.

An A scale dword pairs two 16-row tiles (op_sel selects the half, a trace-time immediate): with an odd ``RT`` the
second m-block of a launch starts on an odd tile, so the dwords are re-paired at the first MFMA that uses them (bytes
0/2 from the odd half of group p, bytes 1/3 from the even half of group p + 1).

The sums of one output element run in the K order of the three thirds, so the slab bytes are the amd-kernels split-3
decode GEMM's bytes (checked bit for bit against it outside this repository).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.qkv_chain_cooperative import (
    PRELOAD_HINTS,
    W_CACHE_MOD,
    _lookahead,
    _mfma,
    _pack8,
    _pin_accumulators,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

N, K = 2560, 6144
KS = 3  # K thirds (split-K workgroups)

# (rows, row tiles RT, column tiles NI, non-temporal slab stores, interleave one W step with its A fragments).
# Measured on the TP4 QKV shape: 65-80 rows one 80-row m-block of 32-column tiles (240 workgroups per K third);
# 81-96 rows two 48-row m-blocks of 64-column tiles; 97-128 rows two 64-row m-blocks. The slab is read by the next
# launch at once, so the two-block rows store it non-temporally (no dirty lines parked in the writer's L2).
ROWS = (65, 128)
_ROWS_TABLE = (
    (80, 5, 2, False, False),
    (96, 3, 4, True, False),
    (128, 4, 4, True, True),
)


def supported(m: int) -> bool:
    return ROWS[0] <= m <= ROWS[1]


def row_config(m: int) -> tuple[int, int, bool, bool]:
    """(row tile RT, column tiles NI, non-temporal stores, interleave) of an ``m``-row decode batch."""
    assert supported(m), m
    for upper, rt, ni, nt, il in _ROWS_TABLE:
        if m <= upper:
            return rt, ni, nt, il
    raise AssertionError(m)


def code_rows(m: int) -> int:
    """Rows of (16, 16)-layout activation codes the launch reads: whole m-blocks (``mxfp8_quantize_shuffled`` pads
    the codes to this and the scales to a multiple of 32 rows)."""
    rt = row_config(m)[0]
    bm = 16 * rt
    return (m + bm - 1) // bm * bm


@functools.cache
def compile_mxfp8_gemm_decode_splitk(
    *, RT: int, NI: int, OUT_NT: bool, INTERLEAVE: bool
):
    """Kernel of ``RT`` 16-row tiles x ``NI`` 16-column tiles per workgroup (one wave), K split in thirds.
    Grid = ``KS * ceil(M / (16 * RT)) * (N / (16 * NI))`` workgroups of 64 threads."""
    assert K % 256 == 0 and N % (16 * NI) == 0
    KSTEPS = K // 128  # 128-K steps
    assert KSTEPS % KS == 0 and (KSTEPS // KS) % 2 == 0
    SW = KSTEPS // KS  # steps per workgroup (even: starts on a 256-K scale block)
    BM = 16 * RT  # rows per m-block
    NB = N // (16 * NI)
    SCB = K // 256  # scale blocks (64 dwords) per 32-row / 32-column group
    RP = (
        RT + 1
    ) // 2  # 32-row scale groups per workgroup (an odd RT straddles one more)
    CP = NI if NI % 2 else NI // 2  # W scale dwords per step
    LA_W, LA_A = _lookahead(SW, RT, NI, RP, CP)
    NA = SW * RT  # A blocks per workgroup
    assert 1 <= LA_W <= SW and 1 <= LA_A <= NA, (LA_W, LA_A)
    out_cm = W_CACHE_MOD if OUT_NT else 0
    out_kw = {"cache_modifier": out_cm} if out_cm else {}
    assert N * K <= 0xFFFFFFFF

    name = (
        f"mxfp8_gemm_decode_splitk_n{N}_k{K}_rt{RT}_ni{NI}_ks{KS}_la{LA_W}x{LA_A}"
        + ("_ont" if OUT_NT else "")
        + ("_ip" if INTERLEAVE else "")
    )

    # addresses, not tensors (a tensor argument ends the kernarg preload)
    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_out: fx.Int64,
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        l16, q16 = lane % 16, lane // 16
        kz, wg = bx // (n_mb * NB), bx % (n_mb * NB)
        mb, nb = wg // NB, wg % NB
        lane_off = q16 * 256 + l16 * 16  # a lane's 16 B inside a 1 KB block
        k_base = kz * (SW * 2048)  # this workgroup's first block pair (bytes)

        a_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_a, num_records_bytes=n_mb * (BM * K)
        )
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=N * K
        )
        as_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_as, num_records_bytes=a_scale_bytes
        )
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_ws, num_records_bytes=(N // 32) * SCB * 256
        )
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=i32_m * (N * 4 * KS)
        )

        n16 = [nb * NI + ni for ni in range_constexpr(NI)]
        r16 = [mb * RT + rt for rt in range_constexpr(RT)]
        w_dw = [
            (n16[ni] * (16 * K) + k_base + lane_off) // 4 for ni in range_constexpr(NI)
        ]
        a_dw = [
            (r16[rt] * (16 * K) + k_base + lane_off) // 4 for rt in range_constexpr(RT)
        ]
        sc_lane = q16 * 16 + l16
        sc_k = kz * (SW // 2 * 64)  # this workgroup's first 256-K scale block (dwords)
        ws_dw = [
            ((n16[cp if NI % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + sc_lane
            for cp in range_constexpr(CP)
        ]
        if const_expr(RT % 2 == 1):
            # odd RT: address the 32-row groups from the m-block's first tile rounded down to even
            # (r16[0] // 2 floors); the parity mix below re-pairs them
            as_dw = [
                ((r16[0] // 2 + rp) * SCB) * 64 + sc_k + sc_lane
                for rp in range_constexpr(RP)
            ]
        else:
            as_dw = [
                ((r16[2 * rp] // 2) * SCB) * 64 + sc_k + sc_lane
                for rp in range_constexpr(RP)
            ]
        a_par = r16[0] % 2 if RT % 2 == 1 else None  # m-block parity
        # a 16-column tile of an odd count sits in either half of its 32 group: move its bytes (half, half + 2)
        # down to (0, 2) and select half 0
        w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)] if NI % 2 else None

        def load_pair(rsrc, dw, s, nt):
            kw = {"vec_width": 4, "dtype": fx.Int32}
            if const_expr(nt):
                kw["cache_modifier"] = W_CACHE_MOD
            return _pack8(
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048, **kw),
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw),
            )

        def load_scales(rsrc, dws, s, shift):
            vals = [
                fx.Int32(
                    buffer_ops.buffer_load(
                        rsrc,
                        dw,
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(s // 2) * 256,
                    )
                )
                for dw in dws
            ]
            if const_expr(shift is None):
                return vals
            return [v.shrui(sh) for v, sh in zip(vals, shift)]

        def repair_a_scale(vals, p):
            """Odd RT: re-pair scale dword p to the even-parity layout the MFMA immediates assume -- dword p must
            carry block tiles 2p (bytes 0/2) and 2p + 1 (bytes 1/3). An odd-parity block shifts every atom by one
            half: bytes 0/2 come from half 1 of group p, bytes 1/3 from half 0 of group p + 1 (the last pair's
            bytes 1/3 name a tile past the block and are never consumed). Called at the dword's first MFMA, not at
            its load, so the load keeps its full lookahead."""
            nxt = vals[p + 1] if p + 1 < RP else vals[p]
            return (a_par == 1).select(
                (vals[p].shrui(fx.Int32(8)) & fx.Int32(0x00FF00FF))
                | ((nxt & fx.Int32(0x00FF00FF)) << 8),
                vals[p],
            )

        def load_w_step(s):
            bb = [load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)]
            if const_expr(s % 2 == 0):
                return bb, (
                    load_scales(ws_rsrc, ws_dw, s, w_shift),
                    load_scales(as_rsrc, as_dw, s, None),
                )
            return bb, None

        def load_a(t):
            rt, s = t % RT, t // RT  # block t: row tile, step
            return load_pair(a_rsrc, a_dw[rt], s, False)

        # Both operands run ahead of the MFMAs: W by LA_W steps, A by LA_A (step, row tile) blocks in the order the
        # MFMAs consume them; with everything in flight from the start a wave ends when its last block lands, not
        # after one memory latency per step.
        acc = [[None] * NI for _ in range_constexpr(RT)]
        w_q = [None] * SW
        a_q = [None] * NA
        if const_expr(INTERLEAVE):
            for p in range_constexpr(max(LA_W, (LA_A + RT - 1) // RT)):
                if const_expr(p < LA_W):
                    w_q[p] = load_w_step(p)
                    fx.rocdl.sched_barrier(0)
                for rt in range_constexpr(RT):
                    if const_expr(p * RT + rt < LA_A):
                        a_q[p * RT + rt] = load_a(p * RT + rt)
                fx.rocdl.sched_barrier(0)
        else:
            for s in range_constexpr(LA_W):
                w_q[s] = load_w_step(s)
            for t in range_constexpr(LA_A):
                a_q[t] = load_a(t)
        sw = sa = sa_raw = None
        for s in range_constexpr(SW):
            if const_expr(s + LA_W < SW):
                w_q[s + LA_W] = load_w_step(s + LA_W)
            bb, sc = w_q[s]
            if const_expr(s % 2 == 0):
                sw, sa = sc
                if const_expr(RT % 2 == 1):
                    sa_raw, sa = sa, [None] * RP  # repaired at first use (both steps)
            for rt in range_constexpr(RT):
                t = s * RT + rt
                if const_expr(t + LA_A < NA):
                    a_q[t + LA_A] = load_a(t + LA_A)
                if const_expr(RT % 2 == 1 and sa[rt // 2] is None):
                    sa[rt // 2] = repair_a_scale(sa_raw, rt // 2)
                a_op, sa_v = a_q[t], sa[rt // 2]
                a_q[t] = None
                for ni in range_constexpr(NI):
                    acc[rt][ni] = _mfma(
                        bb[ni],
                        a_op,
                        acc[rt][ni],
                        sw[ni if NI % 2 else ni // 2],
                        sa_v,
                        0 if NI % 2 else ni % 2,
                        rt % 2,
                        s % 2,
                    )
            w_q[s] = None

        flat = _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NI)])
        for i in range_constexpr(RT * NI):
            rt, ni = i // NI, i % NI
            row = mb * BM + rt * 16 + l16
            col = (nb * NI + ni) * 16 + q16 * 4
            buffer_ops.buffer_store(
                Vec(flat[i]).bitcast(fx.Int32),
                out_rsrc,
                ((kz * i32_m + row) * N + col) * 4,
                offset_is_bytes=True,
                mask=row < i32_m,
                **out_kw,
            )

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_w: fx.Int64,
        arg_as: fx.Int64,
        arg_ws: fx.Int64,
        arg_out: fx.Int64,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, n_mb, arg_w, arg_ws, arg_as, arg_a, arg_out
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(64, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch


def mxfp8_gemm_decode_splitk(
    activation: torch.Tensor,  # e4m3 codes, (16, 16) layout, ``code_rows(m)`` rows
    activation_scale: torch.Tensor,  # AITER-shuffled E8M0 scales [ceil(rows / 32) * 32, K / 32]
    weight: torch.Tensor,  # [2560, 6144] preshuffled (16, 16) e4m3 codes
    weight_scale: torch.Tensor,  # its AITER-shuffled E8M0 scales
    m: int,
    out: torch.Tensor | None = None,  # [3, m, 2560] fp32
) -> torch.Tensor:
    """The three K-third fp32 partial sums of ``A @ W^T`` for 65-128 rows: ``out[z]`` is third ``z``. The caller sums
    ``out[0] + out[1] + out[2]`` (in that order; ``qkv_reduce_insert`` does)."""
    assert supported(m), f"{m} rows outside {ROWS}"
    assert tuple(weight.shape) == (N, K) and activation.shape[1] == K
    rows = code_rows(m)
    assert activation.is_contiguous() and weight.is_contiguous()
    assert activation.shape[0] >= rows, (activation.shape, rows)
    assert activation_scale.numel() >= (rows + 31) // 32 * 32 * (K // 32)
    assert weight_scale.numel() == N * (K // 32)
    rt, ni, nt, il = row_config(m)
    if out is None:
        out = torch.empty((KS, m, N), dtype=torch.float32, device=activation.device)
    assert out.dtype == torch.float32 and out.is_contiguous()
    assert out.shape == (KS, m, N)
    n_mb = (m + 16 * rt - 1) // (16 * rt)
    grid = KS * n_mb * (N // (16 * ni))
    _run_compiled(
        compile_mxfp8_gemm_decode_splitk(RT=rt, NI=ni, OUT_NT=nt, INTERLEAVE=il),
        activation.data_ptr(),
        weight.data_ptr(),
        activation_scale.data_ptr(),
        weight_scale.data_ptr(),
        out.data_ptr(),
        m,
        activation_scale.numel() * activation_scale.element_size(),
        n_mb,
        grid,
        torch.cuda.current_stream(),
    )
    return out

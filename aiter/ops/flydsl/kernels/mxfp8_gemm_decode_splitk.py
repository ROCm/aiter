# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels gemm/mxfp8_common/mxfp8_gemm_decode.py (the decode MXFP8 GEMM, its split-K partial-slab
# mode and the tile / lookahead choices measured for the MiniMax-M3.1 TP4 QKV projection N 2560 x K 6144).
"""gfx950 MXFP8 x MXFP8 split-K decode GEMM on the public row-major activation, fp32 partial slabs out.

    slab[z, m, n] = sum_{k in third z of K} A[m, k] * W[n, k]    (E4M3 codes, per-32 E8M0 scales)

A is the plain quantizer's output (``per_group_quant_hip``: row-major codes ``[M, K]``, row-major E8M0 scales
``[M, K / 32]``, no padding); W is ``shuffle_weight(w, layout=(16, 16))`` with its scales in ``shuffle_scale_a16w4``
order. No LDS staging of either operand: a W block is one contiguous 1 KB ``buffer_load`` (lane (klane, r) gets K bytes
``[16*klane, +16)`` of column r of a 64-K block), an A block the same 16 B per lane read from the lane's own row (rows
past M read as zero through the buffer bound). Two blocks (K +0 and +64) make one 8-VGPR operand of
``v_mfma_scale_f32_16x16x128_f8f6f4``. A workgroup is one wave over ``RT`` 16-row tiles and ``NI`` 16-column tiles of one
K third; W (non-temporal loads) and A blocks are issued ahead of the MFMAs as far as the registers allow. A scale dword
pairs two 16-row tiles x two 128-K steps (op_sel / op_sel_hi pick the byte, trace-time immediates): it is gathered from
the two rows' 8-byte K-group runs and packed in registers at its first MFMA, so the loads keep their lookahead. The fp32
slabs ``[KS, M, N]`` are summed in order and rounded to bf16 by the caller (``reduce_slabs`` of the GEMM backend, or a
fused epilogue).

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
KS = 3  # default split-K factor: K thirds (a kernel parameter, see KS_CHOICES)
# split-K factors valid for K = 6144: a part holds KSTEPS / ks 128-K steps, an even count (it starts on a 256-K scale block)
KS_CHOICES = (1, 2, 3, 4, 6)

# (rows, row tiles RT, column tiles NI, non-temporal slab stores, interleave one W step with its A fragments): the tile
# with the fastest whole QKV front (this GEMM + the reduce-insert) per row range, ks = 3, over every (RT, NI, nt, il)
# on the TP4 QKV shape N 2560 x K 6144. 65-96 rows one 48-row m-block of 64-column tiles; 97-128 rows two 64-row
# m-blocks. The slab is read by the next launch at once, so it is stored non-temporally (no dirty lines parked in the
# writer's L2).
ROWS = (65, 128)
_ROWS_TABLE = (
    (96, 3, 4, True, True),
    (120, 4, 4, True, False),
    (128, 4, 4, True, True),
)


def supported(m: int) -> bool:
    return ROWS[0] <= m <= ROWS[1]


def row_config(m: int) -> tuple[int, int, bool, bool]:
    """(row tile RT, column tiles NI, non-temporal stores, interleave) of an ``m``-row decode batch."""
    if not (supported(m)):
        raise ValueError(m)
    for upper, rt, ni, nt, il in _ROWS_TABLE:
        if m <= upper:
            return rt, ni, nt, il
    raise AssertionError(m)


@functools.cache
def compile_mxfp8_gemm_decode_splitk(
    *,
    RT: int,
    NI: int,
    OUT_NT: bool,
    INTERLEAVE: bool,
    N: int = N,
    K: int = K,
    KS: int = KS,
):
    """Kernel of ``RT`` 16-row tiles x ``NI`` 16-column tiles per workgroup (one wave), K split in ``KS`` parts.
    Grid = ``KS * ceil(M / (16 * RT)) * (N / (16 * NI))`` workgroups of 64 threads."""
    if not (K % 256 == 0 and N % (16 * NI) == 0):
        raise ValueError("qkv front: need K % 256 == 0 and N % (16 * NI) == 0")
    KSTEPS = K // 128  # 128-K steps
    if not (KSTEPS % KS == 0 and (KSTEPS // KS) % 2 == 0):
        raise ValueError("qkv front: need KSTEPS % KS == 0 and (KSTEPS // KS) % 2 == 0")
    SW = KSTEPS // KS  # steps per workgroup (even: starts on a 256-K scale block)
    BM = 16 * RT  # rows per m-block
    NB = N // (16 * NI)
    SCB = K // 256  # scale blocks (64 dwords) per 32-row / 32-column group
    RP = (
        RT + 1
    ) // 2  # 32-row scale groups per workgroup (an odd RT straddles one more)
    CP = NI if NI % 2 else NI // 2  # W scale dwords per step
    # A scales are held raw: 4 dwords (two rows x two K steps) per packed dword
    LA_W, LA_A = _lookahead(SW, RT, NI, RP * 4, CP)
    NA = SW * RT  # A blocks per workgroup
    if not (1 <= LA_W <= SW and 1 <= LA_A <= NA):
        raise ValueError(f"{(LA_W, LA_A)}")
    out_cm = W_CACHE_MOD if OUT_NT else 0
    out_kw = {"cache_modifier": out_cm} if out_cm else {}
    if not (N * K <= 0xFFFFFFFF):
        raise ValueError("qkv front: need N * K <= 0xFFFFFFFF")

    name = (
        f"mxfp8_gemm_decode_splitk_n{N}_k{K}_rt{RT}_ni{NI}_ks{KS}_la{LA_W}x{LA_A}"
        + ("_ont" if OUT_NT else "")
        + ("_ip" if INTERLEAVE else "")
    )

    # addresses, not tensors (a tensor argument ends the kernarg preload)
    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        i32_m: fx.Int32,
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
        k_base = kz * (SW * 2048)  # this workgroup's first W block pair (bytes)

        a_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_a, num_records_bytes=i32_m * K
        )
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=N * K
        )
        as_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_as, num_records_bytes=i32_m * (K // 32)
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
        # row-major codes: lane (q16, l16) -> row 16 * tile + l16, K bytes [16 * q16, +16) of a 64-K block
        a_dw = [
            ((r16[rt] * 16 + l16) * K + kz * (SW * 128) + q16 * 16) // 4
            for rt in range_constexpr(RT)
        ]
        sc_lane = q16 * 16 + l16
        sc_k = kz * (SW // 2 * 64)  # this workgroup's first 256-K scale block (dwords)
        ws_dw = [
            ((n16[cp if NI % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + sc_lane
            for cp in range_constexpr(CP)
        ]
        # per 32-row pair rp (local tiles 2 rp, 2 rp + 1) the byte offsets of the two rows' 8 K groups of a 256-K
        # block (an 8-byte read: step 0's four groups, then step 1's)
        as_dw = [
            ((mb * BM + 32 * rp + l16) * (K // 32) + kz * (SW * 4)) // 4
            for rp in range_constexpr(RP)
        ]
        # a 16-column tile of an odd count sits in either half of its 32 group: move its bytes (half, half + 2)
        # down to (0, 2) and select half 0
        w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)] if NI % 2 else None

        def load_pair(rsrc, dw, s, nt, step=2048, half=1024):
            # a step's two 64-K blocks: 1 KB apart in the (16, 16) layout, 64 B apart in a row-major row
            kw = {"vec_width": 4, "dtype": fx.Int32}
            if const_expr(nt):
                kw["cache_modifier"] = W_CACHE_MOD
            return _pack8(
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * step, **kw),
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * step + half, **kw),
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

        def load_scales_row(s):
            """Row-major A scales of 256-K block s // 2: per 32-row pair the two rows' 8 K-group bytes (two dwords:
            step 0's groups, step 1's), unpacked."""
            out = []
            for dw in as_dw:
                pair = []
                for hi in (0, 1):
                    v = Vec(
                        buffer_ops.buffer_load(
                            as_rsrc,
                            dw + hi * (16 * (K // 32) // 4),
                            vec_width=2,
                            dtype=fx.Int32,
                            soffset_bytes=(s // 2) * 8,
                        )
                    )
                    pair.append((fx.Int32(v[0]), fx.Int32(v[1])))
                out.append(pair)
            return out

        def pack_a_scale_row(raw, p):
            """Scale dword p of the 32-row pair p: bytes (step 0, row l16), (step 0, row l16 + 16), (step 1, row l16),
            (step 1, row l16 + 16), each the byte of this lane's K quad (q16) -- the layout the MFMA op_sel
            immediates address (op_sel = row half, op_sel_hi = step)."""
            (lo0, lo1), (hi0, hi1) = raw[p]
            sh = q16 * 8
            m8 = fx.Int32(0xFF)
            return (
                (lo0.shrui(sh) & m8)
                | ((hi0.shrui(sh) & m8) << 8)
                | ((lo1.shrui(sh) & m8) << 16)
                | (hi1.shrui(sh) << 24)
            )

        def load_w_step(s):
            bb = [load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)]
            if const_expr(s % 2 == 0):
                return bb, (
                    load_scales(ws_rsrc, ws_dw, s, w_shift),
                    load_scales_row(s),
                )
            return bb, None

        def load_a(t):
            rt, s = t % RT, t // RT  # block t: row tile, step
            return load_pair(a_rsrc, a_dw[rt], s, False, 128, 64)

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
                sa_raw, sa = sa, [None] * RP  # packed at first use (both steps)
            for rt in range_constexpr(RT):
                t = s * RT + rt
                if const_expr(t + LA_A < NA):
                    a_q[t + LA_A] = load_a(t + LA_A)
                if const_expr(sa[rt // 2] is None):
                    sa[rt // 2] = pack_a_scale_row(sa_raw, rt // 2)
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
        n_mb: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(i32_m, n_mb, arg_w, arg_ws, arg_as, arg_a, arg_out).launch(
            grid=(fx.Int64(grid), 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch


def mxfp8_gemm_decode_splitk(
    activation: torch.Tensor,  # [m, K] e4m3 codes, row-major (per_group_quant_hip)
    activation_scale: torch.Tensor,  # [m, K / 32] E8M0, row-major, unshuffled
    weight: torch.Tensor,  # [N, K] preshuffled (16, 16) e4m3 codes
    weight_scale: torch.Tensor,  # its shuffle_scale_a16w4 E8M0 scales (N * K / 32 bytes)
    cfg: (
        tuple[int, int, bool, bool] | None
    ) = None,  # (RT, NI, non-temporal stores, interleave); None = row table
    out: torch.Tensor | None = None,  # [ks, m, N] fp32
    ks: int = KS,
) -> torch.Tensor:
    """The ``ks`` K-part fp32 partial sums of ``A @ W^T``: ``out[z]`` is part ``z`` (K split in ``ks`` equal runs); the
    caller sums ``out[0] + out[1] + ...`` in that order. N and K come from the operands (K a multiple of 768 so a part
    is a whole number of 256-K scale blocks for every ks in KS_CHOICES); with ``cfg`` None the row table needs
    65 <= m <= 128 and N % 64 == 0."""
    if ks not in KS_CHOICES:
        raise ValueError(f"mxfp8_gemm_decode_splitk: ks {ks} not in {KS_CHOICES}")
    m, k = activation.shape
    n = weight.shape[0]
    if not (tuple(weight.shape) == (n, k) and k % 768 == 0):
        raise ValueError(f"{(tuple(weight.shape), k)}")
    if not (activation.is_contiguous() and weight.is_contiguous()):
        raise ValueError(
            "qkv front: need activation.is_contiguous() and weight.is_contiguous()"
        )
    if not (activation_scale.is_contiguous()):
        raise ValueError("qkv front: need activation_scale.is_contiguous()")
    if not (
        activation_scale.numel() == m * (k // 32)
        and weight_scale.numel() == n * (k // 32)
    ):
        raise ValueError(
            "qkv front: need activation_scale.numel() == m * (k // 32) and weight_scale.numel() == n * ( k // 32 )"
        )
    rt, ni, nt, il = row_config(m) if cfg is None else cfg
    if not (n % (16 * ni) == 0):
        raise ValueError("qkv front: need n % (16 * ni) == 0")
    if out is None:
        out = torch.empty((ks, m, n), dtype=torch.float32, device=activation.device)
    if not (
        out.dtype == torch.float32 and out.is_contiguous() and out.shape == (ks, m, n)
    ):
        raise ValueError(
            "qkv front: need out.dtype == torch.float32 and out.is_contiguous() and out.shape == (ks, m, n)"
        )
    n_mb = (m + 16 * rt - 1) // (16 * rt)
    grid = ks * n_mb * (n // (16 * ni))
    dev = activation.device
    with torch.cuda.device(dev):
        _run_compiled(
            compile_mxfp8_gemm_decode_splitk(
                RT=rt, NI=ni, OUT_NT=nt, INTERLEAVE=il, N=n, K=k, KS=ks
            ),
            activation.data_ptr(),
            weight.data_ptr(),
            activation_scale.data_ptr(),
            weight_scale.data_ptr(),
            out.data_ptr(),
            m,
            n_mb,
            grid,
            torch.cuda.current_stream(dev),
            specialization_key=("dev", dev.index),
        )
    return out

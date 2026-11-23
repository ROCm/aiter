# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AttnRes training kernels (forward and backward) for gfx1250.

Each token mixes ``L = K + 1`` candidates, the ``K`` banked block representations
``bank[t, 0..K)`` and the intra-block ``prefix[t]``::

    r_l = rsqrt(mean(v_l ** 2) + eps)          z_l = r_l * <v_l, q * w>
    p   = softmax(scale * z)                   o   = sum_l p_l * v_l
    y   = o * rsqrt(mean(o ** 2) + out_eps) * w_o      (optional output RMSNorm)

The forward keeps only per-token scalars (``z``, ``r``, the log-sum-exp and the
output rstd) for the backward, which rebuilds ``o`` from the candidates. One
workgroup owns one row; the candidates stream through registers under an online
softmax, with the next candidate's loads issued before the current reduction.
The backward walks tokens in a grid-stride loop and sums the ``q * w`` and
``w_o`` gradients over its tokens in registers, so each workgroup writes one fp32
partial row instead of one row per token.
"""

# Do not add ``from __future__ import annotations``: FlyDSL inspects annotations
# at trace time.

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .tensor_shim import buf_scalar_load

WARP_SIZE = 32  # gfx1250
_VEC = 8  # bf16 per 128-bit buffer access
_F32_VEC = 4  # fp32 per 128-bit buffer access
_LOG2E = math.log2(math.e)


_MAX_TILES = 4  # 128-bit tiles per thread; each row costs 4 * _MAX_TILES fp32 VGPRs
_MAX_THREADS = 1024


def pick_threads(hidden_size: int) -> int:
    """Fewest whole waves that split the row into at most ``_MAX_TILES`` tiles per thread.

    The backward holds about five fp32 rows per thread, and gfx1250 has 1024 VGPRs
    per SIMD lane to share between its waves, so the per-thread row slice sets the
    occupancy. Kimi-K3's 7168 runs 224 threads (7 waves) of 4 tiles.
    """
    if hidden_size <= 0 or hidden_size % _VEC:
        raise ValueError(
            f"hidden size {hidden_size} must be a positive multiple of {_VEC}"
        )
    num_vec = hidden_size // _VEC
    for threads in range(WARP_SIZE, _MAX_THREADS + 1, WARP_SIZE):
        if num_vec % threads == 0 and num_vec // threads <= _MAX_TILES:
            return threads
    raise ValueError(f"hidden size {hidden_size} has no supported thread split")


def _pin(values):
    """Materialize ``values`` in VGPRs here.

    FlyDSL has no scheduling barrier for values. Without one LLVM sinks every
    online-softmax update to its last use and keeps all candidate rows live: the
    K = 8 forward allocates 591 VGPRs and the backward 952, one wave per SIMD.
    """
    llvm.inline_asm(
        None,
        [value.ir_value() for value in values],
        "",
        ",".join("v" for _ in values),
        has_side_effects=True,
    )


def _row_helpers(tid, threads, tiles, fm):
    """Per-thread row access: thread ``tid`` owns 128-bit tiles ``tid + i * threads``."""
    bf16_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    f32x1_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    def tile_index(tile):
        return tid + tile * threads

    def issue_bf16(row):
        div = fx.logical_divide(row, fx.make_layout(_VEC, 1))
        regs = []
        for tile in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            fx.copy(bf16_atom, fx.slice(div, (None, tile_index(tile))), reg)
            regs.append(reg)
        return regs

    def take_f32(regs):
        return [fx.memref_load_vec(reg).to(fx.Float32) for reg in regs]

    def load_f32(row):
        div = fx.logical_divide(row, fx.make_layout(_F32_VEC, 1))
        values = []
        for tile in range_constexpr(tiles):
            halves = []
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                fx.copy(
                    f32_atom, fx.slice(div, (None, 2 * tile_index(tile) + half)), reg
                )
                halves.append(fx.memref_load_vec(reg))
            values.append(halves[0].shuffle(halves[1], list(range(_VEC))))
        return values

    def store_bf16(row, values):
        div = fx.logical_divide(row, fx.make_layout(_VEC, 1))
        for tile in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            fx.memref_store_vec(values[tile].to(fx.BFloat16), reg)
            fx.copy(bf16_atom, reg, fx.slice(div, (None, tile_index(tile))))

    def store_f32(row, values):
        div = fx.logical_divide(row, fx.make_layout(_F32_VEC, 1))
        for tile in range_constexpr(tiles):
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                lanes = list(range(half * _F32_VEC, (half + 1) * _F32_VEC))
                fx.memref_store_vec(values[tile].shuffle(values[tile], lanes), reg)
                fx.copy(
                    f32_atom, reg, fx.slice(div, (None, 2 * tile_index(tile) + half))
                )

    def load_scalar(row, index):
        div = fx.logical_divide(row, fx.make_layout(1, 1))
        reg = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy(f32x1_atom, fx.slice(div, (None, index)), reg)
        return fx.memref_load_vec(reg)[0]

    def store_scalar(row, index, value):
        div = fx.logical_divide(row, fx.make_layout(1, 1))
        reg = fx.make_rmem_tensor(1, fx.Float32)
        fx.memref_store_vec(fx.full(1, value, fx.Float32), reg)
        fx.copy(f32x1_atom, reg, fx.slice(div, (None, index)))

    def dot(a, b):
        total = (a[0] * b[0]).reduce(ReductionOp.ADD, fastmath=fm)
        for tile in range_constexpr(1, tiles):
            total = total + (a[tile] * b[tile]).reduce(ReductionOp.ADD, fastmath=fm)
        return total

    return (
        issue_bf16,
        take_f32,
        load_f32,
        store_bf16,
        store_f32,
        load_scalar,
        store_scalar,
        dot,
    )


def _uniform(value):
    """A wave-uniform fp32 value moved to an SGPR, so it costs no VGPR."""
    bits = value.bitcast(fx.Int32)
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, bits)).bitcast(fx.Float32)


def _load_uniform(buf, index):
    """fp32 ``buf[index]`` for a wave-uniform ``index``, loaded straight to an SGPR."""
    return fx.Int32(buf_scalar_load(buf, index)).bitcast(fx.Float32)


def build_attn_res_fwd(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    eps: float,
    out_eps: float,
    scale: float,
):
    threads = pick_threads(hidden_size)
    tiles = hidden_size // _VEC // threads
    n_waves = threads // WARP_SIZE
    n_src = num_bank + 1
    red = 2  # values per block reduction
    inv_h = 1.0 / hidden_size
    name = f"attn_res_train_fwd_h{hidden_size}_k{num_bank}" + (
        "_onorm" if output_norm else ""
    )

    @fx.struct
    class SharedStorage:
        partials: fx.Array[fx.Float32, 2 * red * n_waves, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def fwd_kernel(
        bank: fx.Tensor,
        prefix: fx.Tensor,
        qw: fx.Tensor,
        out_weight: fx.Tensor,
        y: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        token = fx.Int32(fx.block_idx.x)
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        fm = arith.FastMathFlags.fast
        s_part = (
            fx.SharedAllocator()
            .allocate(SharedStorage)
            .peek()
            .partials.view(fx.make_layout(2 * red * n_waves, 1))
        )
        (
            issue_bf16,
            take_f32,
            load_f32,
            store_bf16,
            _store_f32,
            _load_scalar,
            store_scalar,
            dot,
        ) = _row_helpers(tid, threads, tiles, fm)

        def block_sum(values, buf):
            totals = []
            for value in values:
                for shift in range_constexpr(4, -1, -1):
                    value = value + value.shuffle_xor(
                        fx.Int32(1 << shift), fx.Int32(WARP_SIZE)
                    )
                totals.append(value)
            if const_expr(n_waves == 1):
                return [_uniform(total) for total in totals]
            base = buf * red * n_waves
            if lane == 0:
                for i in range_constexpr(len(totals)):
                    fx.memref_store(totals[i], s_part, base + i * n_waves + wave)
            fx.gpu.barrier()
            sums = []
            for i in range_constexpr(len(totals)):
                total = fx.memref_load(s_part, base + i * n_waves)
                for w in range_constexpr(1, n_waves):
                    total = total + fx.memref_load(s_part, base + i * n_waves + w)
                sums.append(_uniform(total))
            return sums

        prefix_buf = fx.rocdl.make_buffer_tensor(prefix)
        if const_expr(num_bank):
            bank_buf = fx.rocdl.make_buffer_tensor(bank)

        def issue_source(src):
            if const_expr(src < num_bank):
                row = fx.slice(bank_buf, (token, src, None))
            else:
                row = fx.slice(prefix_buf, (token, None))
            return issue_bf16(row)

        logit_row = fx.slice(fx.rocdl.make_buffer_tensor(logit), (token, None))
        rstd_row = fx.slice(fx.rocdl.make_buffer_tensor(rstd), (token, None))

        pending = issue_source(0)
        qw_vals = load_f32(fx.rocdl.make_buffer_tensor(qw))
        acc = None
        run_max = None
        denom = None
        for src in range_constexpr(n_src):
            vals = take_f32(pending)
            if const_expr(src + 1 < n_src):
                pending = issue_source(src + 1)
            sumsq, qdot = block_sum([dot(vals, vals), dot(vals, qw_vals)], src % 2)
            r = fmath.rsqrt(sumsq * inv_h + eps, fastmath=fm)
            z = qdot * r
            s = z * scale
            if const_expr(src == 0):
                run_max = s
                denom = fx.Float32(1.0)
                acc = vals
            else:
                new_max = run_max.maximumf(s)
                old_scale = fmath.exp2((run_max - new_max) * _LOG2E, fastmath=fm)
                new_scale = fmath.exp2((s - new_max) * _LOG2E, fastmath=fm)
                denom = denom * old_scale + new_scale
                acc = [
                    acc[t] * old_scale + vals[t] * new_scale
                    for t in range_constexpr(tiles)
                ]
                run_max = new_max
            _pin(acc)
            if tid == 0:
                store_scalar(logit_row, src, z)
                store_scalar(rstd_row, src, r)

        inv_denom = 1.0 / denom
        out = [acc[t] * inv_denom for t in range_constexpr(tiles)]
        lse_val = run_max + fmath.log(denom, fastmath=fm)
        out_r = fx.Float32(1.0)
        if const_expr(output_norm):
            (out_sumsq,) = block_sum([dot(out, out)], n_src % 2)
            out_r = fmath.rsqrt(out_sumsq * inv_h + out_eps, fastmath=fm)
            w_out = load_f32(fx.rocdl.make_buffer_tensor(out_weight))
            out = [out[t] * out_r * w_out[t] for t in range_constexpr(tiles)]
        store_bf16(fx.slice(fx.rocdl.make_buffer_tensor(y), (token, None)), out)
        if tid == 0:
            store_scalar(fx.rocdl.make_buffer_tensor(lse), token, lse_val)
            if const_expr(output_norm):
                store_scalar(fx.rocdl.make_buffer_tensor(out_rstd), token, out_r)

    @flyc.jit
    def launch_fwd(
        bank: fx.Tensor,
        prefix: fx.Tensor,
        qw: fx.Tensor,
        out_weight: fx.Tensor,
        y: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        fwd_kernel(bank, prefix, qw, out_weight, y, logit, rstd, lse, out_rstd).launch(
            grid=(num_tokens, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch_fwd


def build_attn_res_bwd(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    scale: float,
):
    threads = pick_threads(hidden_size)
    tiles = hidden_size // _VEC // threads
    n_waves = threads // WARP_SIZE
    n_src = num_bank + 1
    red = 2  # values per block reduction
    inv_h = 1.0 / hidden_size
    name = f"attn_res_train_bwd_h{hidden_size}_k{num_bank}" + (
        "_onorm" if output_norm else ""
    )

    # The w_o gradient sums in LDS rather than in a loop-carried register row,
    # which would be live across pass 2 and cost 32 VGPRs at Kimi-K3's width.
    dwo_elems = hidden_size if output_norm else _F32_VEC

    @fx.struct
    class SharedStorage:
        partials: fx.Array[fx.Float32, 2 * red * n_waves, 16]
        dwo: fx.Array[fx.Float32, dwo_elems, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def bwd_kernel(
        dy: fx.Tensor,
        bank: fx.Tensor,
        prefix: fx.Tensor,
        qw: fx.Tensor,
        out_weight: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        dbank: fx.Tensor,
        dprefix: fx.Tensor,
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        group = fx.Int32(fx.block_idx.x)
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        fm = arith.FastMathFlags.fast
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_part = lds.partials.view(fx.make_layout(2 * red * n_waves, 1))
        s_dwo = fx.logical_divide(
            lds.dwo.view(fx.make_layout(dwo_elems, 1)), fx.make_layout(_F32_VEC, 1)
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float32)
        (
            issue_bf16,
            take_f32,
            load_f32,
            store_bf16,
            store_f32,
            _load_scalar,
            _store_scalar,
            dot,
        ) = _row_helpers(tid, threads, tiles, fm)

        # Each thread owns the same elements of s_dwo as of its rows, so the
        # accumulation needs no barrier.
        def load_dwo(tile):
            halves = []
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                index = 2 * (tid + tile * threads) + half
                fx.copy(lds_atom, fx.slice(s_dwo, (None, index)), reg)
                halves.append(fx.memref_load_vec(reg))
            return halves[0].shuffle(halves[1], list(range(_VEC)))

        def store_dwo(tile, value):
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                lanes = list(range(half * _F32_VEC, (half + 1) * _F32_VEC))
                fx.memref_store_vec(value.shuffle(value, lanes), reg)
                index = 2 * (tid + tile * threads) + half
                fx.copy(lds_atom, reg, fx.slice(s_dwo, (None, index)))

        if const_expr(output_norm):
            for tile in range_constexpr(tiles):
                store_dwo(tile, fx.Vector.filled(_VEC, 0.0, fx.Float32))

        def block_sum(values, buf):
            totals = []
            for value in values:
                for shift in range_constexpr(4, -1, -1):
                    value = value + value.shuffle_xor(
                        fx.Int32(1 << shift), fx.Int32(WARP_SIZE)
                    )
                totals.append(value)
            if const_expr(n_waves == 1):
                return [_uniform(total) for total in totals]
            base = buf * red * n_waves
            if lane == 0:
                for i in range_constexpr(len(totals)):
                    fx.memref_store(totals[i], s_part, base + i * n_waves + wave)
            fx.gpu.barrier()
            sums = []
            for i in range_constexpr(len(totals)):
                total = fx.memref_load(s_part, base + i * n_waves)
                for w in range_constexpr(1, n_waves):
                    total = total + fx.memref_load(s_part, base + i * n_waves + w)
                sums.append(_uniform(total))
            return sums

        dy_buf = fx.rocdl.make_buffer_tensor(dy)
        prefix_buf = fx.rocdl.make_buffer_tensor(prefix)
        dprefix_buf = fx.rocdl.make_buffer_tensor(dprefix)
        if const_expr(num_bank):
            bank_buf = fx.rocdl.make_buffer_tensor(bank)
            dbank_buf = fx.rocdl.make_buffer_tensor(dbank)
        logit_buf = fx.rocdl.make_buffer_tensor(logit)
        rstd_buf = fx.rocdl.make_buffer_tensor(rstd)
        lse_buf = fx.rocdl.make_buffer_tensor(lse)
        out_rstd_buf = fx.rocdl.make_buffer_tensor(out_rstd)
        qw_buf = fx.rocdl.make_buffer_tensor(qw)

        def issue_source(token, src):
            if const_expr(src < num_bank):
                row = fx.slice(bank_buf, (token, src, None))
            else:
                row = fx.slice(prefix_buf, (token, None))
            return issue_bf16(row)

        def token_step(token, dqw_acc):
            pending = issue_source(token, 0)
            dy_vals = take_f32(issue_bf16(fx.slice(dy_buf, (token, None))))
            # Per-token scalars are loaded where they are used, straight to SGPRs.
            stat_base = token * n_src
            lse_t = _load_uniform(lse_buf, token)

            def candidate_stats(src):
                z = _load_uniform(logit_buf, stat_base + src)
                p = fmath.exp2((z * scale - lse_t) * _LOG2E, fastmath=fm)
                return z, p

            # Pass 1: rebuild o = sum_l p_l v_l.
            out = None
            for src in range_constexpr(n_src):
                vals = take_f32(pending)
                if const_expr(src + 1 < n_src):
                    pending = issue_source(token, src + 1)
                _, p = candidate_stats(src)
                if const_expr(src == 0):
                    out = [vals[t] * p for t in range_constexpr(tiles)]
                else:
                    out = [out[t] + vals[t] * p for t in range_constexpr(tiles)]
                _pin(out)
            # Issuing pass 2's first row here would hold it across the peak below.
            if const_expr(not output_norm):
                pending = issue_source(token, 0)

            # Output RMSNorm backward and delta = <dL/do, o>, in one reduction:
            # with xhat = o * r_o, <xhat, w_o dy> is r_o * <w_o dy, o>.
            if const_expr(output_norm):
                out_r = _load_uniform(out_rstd_buf, token)
                w_out = load_f32(fx.rocdl.make_buffer_tensor(out_weight))
                for t in range_constexpr(tiles):
                    store_dwo(t, load_dwo(t) + out[t] * (dy_vals[t] * out_r))
                wdy = [w_out[t] * dy_vals[t] for t in range_constexpr(tiles)]
                b, c = block_sum([dot(wdy, out), dot(out, out)], 0)
                c1 = out_r * b * inv_h
                delta = out_r * b - out_r * out_r * c1 * c
                xhat_coef = out_r * c1
                g = [
                    (wdy[t] - out[t] * xhat_coef) * out_r
                    for t in range_constexpr(tiles)
                ]
                _pin(g)
                pending = issue_source(token, 0)
            else:
                g = dy_vals
                (delta,) = block_sum([dot(g, out)], 0)
            qw_vals = load_f32(qw_buf)

            # Pass 2: per-candidate softmax and RMSNorm backward. A single
            # candidate has p = 1 whatever its score, so its score gets no gradient.
            for src in range_constexpr(n_src):
                vals = take_f32(pending)
                if const_expr(src + 1 < n_src):
                    pending = issue_source(token, src + 1)
                if const_expr(n_src == 1):
                    dv = g
                else:
                    (dp,) = block_sum([dot(vals, g)], (src + 1) % 2)
                    z, p = candidate_stats(src)
                    r = _load_uniform(rstd_buf, stat_base + src)
                    dz = (dp - delta) * (p * scale)
                    key_coef = dz * r
                    key_shift = r * z * inv_h
                    dv = [
                        g[t] * p + (qw_vals[t] - vals[t] * key_shift) * key_coef
                        for t in range_constexpr(tiles)
                    ]
                    dqw_acc = [
                        dqw_acc[t] + vals[t] * key_coef for t in range_constexpr(tiles)
                    ]
                    _pin(dqw_acc)
                if const_expr(src < num_bank):
                    store_bf16(fx.slice(dbank_buf, (token, src, None)), dv)
                else:
                    store_bf16(fx.slice(dprefix_buf, (token, None)), dv)
            # The next token's first reduction reuses LDS slot 0.
            fx.gpu.barrier()
            return dqw_acc

        # With one candidate the scores get no gradient and nothing is carried;
        # otherwise the q * w gradient is. A single carried value comes back bare
        # rather than in a list.
        if const_expr(n_src > 1):
            init = [
                fx.Vector.filled(_VEC, 0.0, fx.Float32) for _ in range_constexpr(tiles)
            ]
            for it, state in range(group, num_tokens, num_groups, init=init):
                dqw_acc = token_step(fx.Int32(it), list(state))
                results = yield dqw_acc
            if const_expr(tiles == 1):
                results = [results]
            store_f32(
                fx.slice(fx.rocdl.make_buffer_tensor(part_dqw), (group, None)),
                list(results),
            )
        else:
            for it in range(group, num_tokens, num_groups):
                token_step(fx.Int32(it), [])
        if const_expr(output_norm):
            store_f32(
                fx.slice(fx.rocdl.make_buffer_tensor(part_dwo), (group, None)),
                [load_dwo(t) for t in range_constexpr(tiles)],
            )

    @flyc.jit
    def launch_bwd(
        dy: fx.Tensor,
        bank: fx.Tensor,
        prefix: fx.Tensor,
        qw: fx.Tensor,
        out_weight: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        dbank: fx.Tensor,
        dprefix: fx.Tensor,
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        bwd_kernel(
            dy,
            bank,
            prefix,
            qw,
            out_weight,
            logit,
            rstd,
            lse,
            out_rstd,
            dbank,
            dprefix,
            part_dqw,
            part_dwo,
            num_tokens,
            num_groups,
        ).launch(grid=(num_groups, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch_bwd

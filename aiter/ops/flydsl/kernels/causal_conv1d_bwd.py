# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backward of the varlen causal conv1d with split q/k/v output (FlyDSL).

Forward, per sequence, channel ``c`` and token ``t``:
``pre[t] = bias[c] + sum_k w[c, k] * xe[t - (width - 1) + k]``, ``out = act(pre)``,
where ``xe`` is the sequence prefixed by ``width - 1`` initial-state columns.

Same layout and semantics as the Triton kernel
``aiter.ops.triton._triton_kernels.gated_delta_net.prefill.causal_conv1d_bwd_split_qkv``:
each program owns one ``(sequence, chunk)`` of the chunk schedule and
``block * lanes`` channels, recomputes the pre-activation from ``x``, writes
``dx``, and writes fp32 per-chunk partials of ``dweight`` (and ``dbias``) that
the host sums in a fixed order.

Each thread owns ``lanes`` adjacent channels and walks the chunk plus its
``width - 1`` halo rows in a compile-time unrolled loop, ``prefetch`` token rows
of x and of the output gradient loaded ahead of their use. The last
``width - 1`` rows of ``xe`` and of the pre-activation gradient stay in
registers. The q/k/v source of the output gradient is picked once per thread,
outside the token loop. Rows past a sequence's live length (and every row of a
padded sequence) read clamped, in-bounds addresses and are selected to zero, so
NaN in those rows cannot reach any gradient. Offsets are 64-bit.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr

_DEFAULT_STREAM = fx.Stream(None)

_ELEMENT = {
    torch.bfloat16: fx.BFloat16,
    torch.float16: fx.Float16,
    torch.float32: fx.Float32,
}


@functools.lru_cache(maxsize=128)
def create_causal_conv1d_bwd_kernel(
    dim: int,
    k_dim: int,
    width: int,
    x_unit: int,
    g_units: tuple,
    ws: tuple,
    ss: tuple,
    has_bias: bool,
    silu: bool,
    has_initial: bool,
    has_indices: bool,
    pad_slot: int | None,
    block: int,
    tokens: int,
    prefetch: int,
    lanes: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
):
    """Launcher for one configuration.

    ``x_unit`` / ``g_units`` give the unit-stride axis (0 = channel, 1 = token)
    of x (and dx) and of dq, dk, dv. Weight and state strides are compiled in;
    the token-count-dependent strides are runtime arguments.
    """
    element = _ELEMENT[dtype]
    state_element = _ELEMENT[state_dtype]
    halo = width - 1
    steps = tokens + halo
    channels = block * lanes
    n_rows = width + (1 if has_bias else 0)

    @flyc.jit
    def load_lanes(tensor: fx.Tensor, offset: fx.Int64, stride: fx.Int64):
        return fx.Vector.from_elements(
            [tensor[offset + lane * stride] for lane in range_constexpr(lanes)]
        )

    @flyc.kernel(known_block_size=[block, 1, 1])
    def causal_conv1d_bwd_kernel(
        x: fx.Tensor,
        w: fx.Tensor,
        bias: fx.Tensor,
        state: fx.Tensor,
        starts: fx.Tensor,
        indices: fx.Tensor,
        initial: fx.Tensor,
        batch_ids: fx.Tensor,
        chunk_ids: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dv: fx.Tensor,
        dx: fx.Tensor,
        partial: fx.Tensor,
        dinit: fx.Tensor,
        x_span: fx.Int64,
        w_span: fx.Int64,
        state_span: fx.Int64,
        q_span: fx.Int64,
        k_span: fx.Int64,
        v_span: fx.Int64,
        dx_span: fx.Int64,
        part_span: fx.Int64,
        dinit_span: fx.Int64,
        x_stride: fx.Int64,
        dx_stride: fx.Int64,
        q_stride: fx.Int64,
        k_stride: fx.Int64,
        v_stride: fx.Int64,
        n_seqs: fx.Int32,
        cu_seqlen: fx.Int32,
        n_programs: fx.Int32,
    ):
        # (channel stride, token stride) of each [*, token] or [token, *] tensor.
        def axis_strides(unit, stride):
            one = fx.Int64(1)
            return (one, stride) if unit == 0 else (stride, one)

        xs0, xs1 = axis_strides(x_unit, x_stride)
        ds0, ds1 = axis_strides(x_unit, dx_stride)
        qs0, qs1 = axis_strides(g_units[0], q_stride)
        ks0, ks1 = axis_strides(g_units[1], k_stride)
        vs0, vs1 = axis_strides(g_units[2], v_stride)

        # Explicit physical offsets below already include the strides.
        def flat(t, span):
            return fx.Tensor(fx.make_view(fx.get_iter(t), fx.make_layout(span, 1)))

        x = flat(x, x_span)
        w = flat(w, w_span)
        state = flat(state, state_span)
        dq = flat(dq, q_span)
        dk = flat(dk, k_span)
        dv = flat(dv, v_span)
        dx = flat(dx, dx_span)
        partial = flat(partial, part_span)
        dinit = flat(dinit, dinit_span)

        pid = fx.block_idx.x
        c = (fx.block_idx.y * block + fx.thread_idx.x) * lanes
        cc = fx.Int64(c)
        seq = fx.Int32(batch_ids[pid])
        chunk = fx.Int64(chunk_ids[pid])
        start = fx.Int64(starts[seq])
        length = fx.Int64(starts[seq + 1]) - start
        c0 = chunk * tokens
        n_tok = (length - c0 < tokens).select(length - c0, fx.Int64(tokens))
        last = length - 1
        slot = fx.Int64(seq)
        if const_expr(has_indices):
            slot = fx.Int64(indices[seq])
        # A padded sequence reads nothing live, so its gradients are exactly zero.
        live = length
        if const_expr(pad_slot is not None):
            live = (slot == pad_slot).select(fx.Int64(0), length)
        n_state = fx.Int64(0)
        if const_expr(has_initial):
            use = (fx.Int32(initial[seq]) != 0) & (live == length)
            n_state = use.select(fx.Int64(halo), fx.Int64(0))
        zeros = fx.Vector.filled(lanes, 0.0, fx.Float32)
        one = fx.Vector.filled(lanes, 1.0, fx.Float32)

        def vsel(cond, a, b):
            return fx.Vector.from_elements(
                [cond.select(a[lane], b[lane]) for lane in range_constexpr(lanes)]
            )

        # Tensors come in as parameters: the frontend must see them as locals of
        # the helper, since its runtime branches store through them.
        def run(g, gc, gs0, gs1, x, w, bias, state, dx, partial, dinit):
            weights = [
                load_lanes(w, cc * ws[0] + j * ws[1], fx.Int64(ws[0])).to(fx.Float32)
                for j in range_constexpr(width)
            ]
            base = zeros
            if const_expr(has_bias):
                base = load_lanes(bias, cc, fx.Int64(1)).to(fx.Float32)
            # xw[j] = xe[c0 - halo + j], oldest first.
            xw = []
            for j in range_constexpr(halo):
                pos = c0 - halo + j
                in_x = (pos >= 0) & (pos < live)
                safe = (pos >= 0).select(pos, fx.Int64(0))
                value = load_lanes(x, cc * xs0 + (start + safe) * xs1, xs0).to(
                    fx.Float32
                )
                value = vsel(in_x, value, zeros)
                if const_expr(has_initial):
                    hvalue = zeros
                    # Keep compile-time and runtime branches separate for FlyDSL.
                    if (pos < 0) & (pos + n_state >= 0):
                        hvalue = load_lanes(
                            state, slot * ss[0] + cc * ss[1] + (pos + halo) * ss[2],
                            fx.Int64(ss[1]),
                        ).to(fx.Float32)  # fmt: skip
                    value = value + hvalue
                xw.append(value)
            # gw[j] = dpre[u - halo + j] of the rows already walked, oldest first.
            gw = [zeros for _ in range_constexpr(halo)]
            dw = [zeros for _ in range_constexpr(width)]
            db = zeros
            # dpre of the first width - 1 rows of the chunk, for dinitial_states.
            g_first = [zeros for _ in range_constexpr(halo if has_initial else 0)]

            def load_group(first):
                """x rows, then output-gradient rows, ``first .. first + prefetch - 1`` of the chunk."""
                xs = []
                gs = []
                for p in range_constexpr(prefetch):
                    pos = c0 + first + p
                    safe = (pos < length).select(pos, last)
                    xs.append(load_lanes(x, cc * xs0 + (start + safe) * xs1, xs0))
                    gs.append(load_lanes(g, gc * gs0 + (start + safe) * gs1, gs0))
                return xs + gs

            def row(i, raw_x, raw_g, xw, gw, dw, db, g_first, dx):
                """Chunk row ``i`` (int64): update the sums, store dx[i - halo]."""
                ok = (c0 + i) < live
                xu = vsel(ok, raw_x.to(fx.Float32), zeros)
                gu = vsel(ok, raw_g.to(fx.Float32), zeros)
                pre = base + weights[halo] * xu
                for j in range_constexpr(halo):
                    pre = pre + weights[j] * xw[j]
                dpre = gu
                if const_expr(silu):
                    sig = one / (one + fx.exp(-pre))
                    dpre = gu * sig * (one + pre * (one - sig))

                gin = vsel(i < n_tok, dpre, zeros)
                dw = [dw[j] + gin * xw[j] for j in range_constexpr(halo)] + [
                    dw[halo] + gin * xu
                ]
                if const_expr(has_bias):
                    db = db + gin
                if const_expr(has_initial):
                    g_first = [
                        vsel(i == k, dpre, g_first[k]) for k in range_constexpr(halo)
                    ]

                # dx[u - halo] = sum_j w[j] * dpre[u - j]
                dxv = weights[0] * dpre
                for j in range_constexpr(1, width):
                    dxv = dxv + weights[j] * gw[halo - j]
                t = i - halo
                if (t >= 0) & (t < n_tok):
                    r = start + c0 + t
                    for lane in range_constexpr(lanes):
                        dx[(cc + lane) * ds0 + r * ds1] = element(dxv[lane])
                return xw[1:] + [xu], gw[1:] + [dpre], dw, db, g_first

            # Fully unrolled: register use grows with the chunk length, so short
            # chunks (block_m 32) do best. A runtime loop over prefetch groups was
            # not faster on gfx950.
            raws = []
            for i in range_constexpr(steps):
                if const_expr(i % prefetch == 0):
                    raws = load_group(fx.Int64(i))
                p = i % prefetch
                xw, gw, dw, db, g_first = row(
                    fx.Int64(i), raws[p], raws[prefetch + p],
                    xw, gw, dw, db, g_first, dx,
                )  # fmt: skip

            # Per-chunk partial sums; the host reduces them over chunks (deterministic).
            part = (fx.Int64(pid) * dim + cc) * n_rows
            for j in range_constexpr(width):
                for lane in range_constexpr(lanes):
                    partial[part + lane * n_rows + j] = fx.Float32(dw[j][lane])
            if const_expr(has_bias):
                for lane in range_constexpr(lanes):
                    partial[part + lane * n_rows + width] = fx.Float32(db[lane])

            # dinit[k] = sum_{j <= k} w[j] * dpre[k - j], from the first chunk only
            if const_expr(has_initial):  # noqa: SIM102
                if (chunk == 0) & (n_state > 0):
                    for k in range_constexpr(halo):
                        di = zeros
                        for j in range_constexpr(k + 1):
                            di = di + weights[j] * g_first[k - j]
                        for lane in range_constexpr(lanes):
                            dinit[(fx.Int64(seq) * dim + cc + lane) * halo + k] = (
                                state_element(di[lane])
                            )

        outs = (x, w, bias, state, dx, partial, dinit)
        if c < dim:
            if c < k_dim:
                run(dq, cc, qs0, qs1, *outs)
            else:
                if c < 2 * k_dim:
                    run(dk, cc - k_dim, ks0, ks1, *outs)
                else:
                    run(dv, cc - 2 * k_dim, vs0, vs1, *outs)

        # dx of the tokens after the last sequence is zero. The end is read on the
        # device, so a reused schedule needs no host sync; the programs share the rows.
        end = fx.Int32(starts[n_seqs])
        if c < dim:
            for r in range(end + pid, cu_seqlen, n_programs):
                for lane in range_constexpr(lanes):
                    dx[(cc + lane) * ds0 + fx.Int64(r) * ds1] = element(0.0)

    @flyc.jit
    def launch(
        x: fx.Tensor,
        w: fx.Tensor,
        bias: fx.Tensor,
        state: fx.Tensor,
        starts: fx.Tensor,
        indices: fx.Tensor,
        initial: fx.Tensor,
        batch_ids: fx.Tensor,
        chunk_ids: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dv: fx.Tensor,
        dx: fx.Tensor,
        partial: fx.Tensor,
        dinit: fx.Tensor,
        x_span: fx.Int64,
        w_span: fx.Int64,
        state_span: fx.Int64,
        q_span: fx.Int64,
        k_span: fx.Int64,
        v_span: fx.Int64,
        dx_span: fx.Int64,
        part_span: fx.Int64,
        dinit_span: fx.Int64,
        x_stride: fx.Int64,
        dx_stride: fx.Int64,
        q_stride: fx.Int64,
        k_stride: fx.Int64,
        v_stride: fx.Int64,
        n_seqs: fx.Int32,
        cu_seqlen: fx.Int32,
        n_programs: fx.Int32,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        causal_conv1d_bwd_kernel(
            x,
            w,
            bias,
            state,
            starts,
            indices,
            initial,
            batch_ids,
            chunk_ids,
            dq,
            dk,
            dv,
            dx,
            partial,
            dinit,
            x_span,
            w_span,
            state_span,
            q_span,
            k_span,
            v_span,
            dx_span,
            part_span,
            dinit_span,
            x_stride,
            dx_stride,
            q_stride,
            k_stride,
            v_stride,
            n_seqs,
            cu_seqlen,
            n_programs,
        ).launch(
            grid=(fx.Int64(n_programs), (dim + channels - 1) // channels, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch

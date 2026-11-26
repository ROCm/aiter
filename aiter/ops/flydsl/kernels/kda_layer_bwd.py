# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Element-wise / row-wise backward kernels of the Kimi Delta Attention layer.

Launch wrappers are in ``../kda_layer_bwd.py``. The conv1d + SiLU stage is not
here: it reuses the Triton ``causal_conv1d_split_qkv_bwd_triton_fn``. Every
tensor is contiguous fp32 and addressed through a flat unit-stride view; the
wrapper guarantees both.

* ``l2norm_bwd``: one wave per row, rows of ``K`` (a multiple of 64).
* ``gate_beta_bwd``: F-decay gate ``g = -exp(A_log) * softplus(z)`` and
  ``beta = sigmoid(b)``, one element per lane.

Factories take the device index only as a cache key: a compiled handle does not
launch on another device.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

_DEFAULT_STREAM = fx.Stream(None)
WAVE = 64


def _flat(t, span):
    return fx.Tensor(fx.make_view(fx.get_iter(t), fx.make_layout(span, 1)))


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-x))


def _wave_sum(v):
    for sh in [32, 16, 8, 4, 2, 1]:
        v = v + v.shuffle_xor(fx.Int32(sh), fx.Int32(WAVE))
    return v


@functools.lru_cache(maxsize=32)
def create_l2norm_bwd_kernel(
    dim: int, eps: float, waves: int, device: int | None = None
):
    """x, dy, dx: [rows, dim]. y = x / max(||x||, eps)."""
    E = dim // WAVE
    block = waves * WAVE

    @flyc.kernel(known_block_size=[block, 1, 1])
    def l2norm_bwd_kernel(
        x: fx.Tensor, dy: fx.Tensor, dx: fx.Tensor, rows: fx.Int64, span: fx.Int64
    ):
        x = _flat(x, span)
        dy = _flat(dy, span)
        dx = _flat(dx, span)
        tid = fx.Int64(fx.thread_idx.x)
        row = fx.Int64(fx.block_idx.x) * waves + tid // WAVE
        lane = tid % WAVE
        safe = (row < rows).select(row, fx.Int64(0))
        offs = [safe * dim + lane + e * WAVE for e in range_constexpr(E)]
        xv = [fx.Float32(x[o]) for o in offs]
        gv = [fx.Float32(dy[o]) for o in offs]
        ss = fx.Float32(0.0)
        dot = fx.Float32(0.0)
        for e in range_constexpr(E):
            ss = ss + xv[e] * xv[e]
            dot = dot + xv[e] * gv[e]
        ss = _wave_sum(ss)
        dot = _wave_sum(dot)
        norm = fx.sqrt(ss)
        clamped = norm < fx.Float32(eps)
        inv = fx.Float32(1.0) / clamped.select(fx.Float32(eps), norm)
        # below eps the norm is a constant, so the projection term vanishes
        proj = clamped.select(fx.Float32(0.0), dot * inv * inv * inv)
        if row < rows:
            for e in range_constexpr(E):
                dx[offs[e]] = gv[e] * inv - xv[e] * proj

    @flyc.jit
    def launch(
        x: fx.Tensor,
        dy: fx.Tensor,
        dx: fx.Tensor,
        rows: fx.Int64,
        span: fx.Int64,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        l2norm_bwd_kernel(x, dy, dx, rows, span).launch(
            grid=((rows + waves - 1) // waves, 1, 1), block=(block, 1, 1), stream=stream
        )

    return launch


@functools.lru_cache(maxsize=32)
def create_gate_beta_bwd_kernel(
    heads: int, head_dim: int, block: int, device: int | None = None
):
    """z, dg, dz, dgg: [N, H*K]; A_log: [H]; b, dbeta, db: [N, H].

    g = -exp(A_log[h]) * softplus(z) (z already includes dt_bias);
    dz = dg * dg/dz, dgg = dg * g (summed into dA_log by the caller);
    db = dbeta * beta * (1 - beta).
    """
    H, K = heads, head_dim

    @flyc.kernel(known_block_size=[block, 1, 1])
    def gate_beta_bwd_kernel(
        z: fx.Tensor,
        a_log: fx.Tensor,
        dg: fx.Tensor,
        braw: fx.Tensor,
        dbeta: fx.Tensor,
        dz: fx.Tensor,
        dgg: fx.Tensor,
        db: fx.Tensor,
        n_gate: fx.Int64,
        n_beta: fx.Int64,
    ):
        z = _flat(z, n_gate)
        dg = _flat(dg, n_gate)
        dz = _flat(dz, n_gate)
        dgg = _flat(dgg, n_gate)
        braw = _flat(braw, n_beta)
        dbeta = _flat(dbeta, n_beta)
        db = _flat(db, n_beta)
        a_log = _flat(a_log, fx.Int64(H))
        i = fx.Int64(fx.block_idx.x * block + fx.thread_idx.x)
        if i < n_gate:
            h = (i // K) % H
            zi = fx.Float32(z[i])
            gi = fx.Float32(dg[i])
            neg_a = -fx.exp(fx.Float32(a_log[h]))
            # softplus(z) = max(z, 0) + log1p(exp(-|z|)), stable for large |z|
            sp = fx.max(zi, fx.Float32(0.0)) + fx.log(
                fx.Float32(1.0) + fx.exp(-fx.max(zi, -zi))
            )
            dz[i] = gi * neg_a * _sigmoid(zi)
            dgg[i] = gi * neg_a * sp
        if i < n_beta:
            bt = _sigmoid(fx.Float32(braw[i]))
            db[i] = fx.Float32(dbeta[i]) * bt * (fx.Float32(1.0) - bt)

    @flyc.jit
    def launch(
        z: fx.Tensor,
        a_log: fx.Tensor,
        dg: fx.Tensor,
        braw: fx.Tensor,
        dbeta: fx.Tensor,
        dz: fx.Tensor,
        dgg: fx.Tensor,
        db: fx.Tensor,
        n_gate: fx.Int64,
        n_beta: fx.Int64,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        gate_beta_bwd_kernel(
            z, a_log, dg, braw, dbeta, dz, dgg, db, n_gate, n_beta
        ).launch(
            grid=((n_gate + block - 1) // block, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch

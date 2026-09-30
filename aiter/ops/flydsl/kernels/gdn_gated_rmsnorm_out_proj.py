# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode kernel: the GDN gated RMSNorm fused into the out_proj GEMM.

A Gated DeltaNet (GDN) layer, as in Qwen3.5 and Qwen3-Next, ends with two ops:

1. ``RMSNormGated`` per value head, with ``norm_before_gate=True`` and a SiLU gate:
   ``y = bf16(x * rsqrt(mean(x^2) + eps) * w * silu(z))``. The mean runs over the
   ``head_dim`` values of one head, and all heads share the weight ``w``.
2. ``out_proj``: ``out = y.flatten(-2) @ W.T`` with a bf16 ``W`` of shape ``[N, K]``,
   where ``K = num_heads * head_dim``.

At decode (1 to 8 tokens) the GEMM is a matrix-vector product that streams ``W`` once
from HBM. This kernel computes ``y`` inside that GEMM, so ``y`` never goes through HBM
and the separate norm kernel is not launched.

How the work is split:

* One wave owns ``cols_per_wave`` consecutive output columns, which are rows of ``W``.
* Lane ``l`` always covers the same 8 values of K in chunk ``c``, from
  ``c * 512 + 8 * l`` to ``c * 512 + 8 * l + 7``. This map does not depend on the
  column, so a lane needs the same 8 values of ``y`` for all of its columns.
* Every wave issues its ``W`` loads first and its ``x`` and ``z`` loads after them, so
  the HBM read of ``W`` starts as early as possible. Loads return in order, so a wave
  starts its part of the norm once its ``W`` loads have arrived. The waves of the
  workgroup compute ``y`` together, one piece of 512 values (one token, one chunk) per
  wave at a time, and write it to LDS as bf16. With
  ``head_dim`` 128, the 16 consecutive lanes that hold one head sum their squares with
  a 16-lane butterfly, so no extra LDS pass is needed for the norm.
* After one barrier every lane reads back its own slice of ``y``, multiplies it with its
  ``W`` values with ``v_dot2_f32_bf16`` and accumulates in fp32. A 64-lane butterfly then
  sums each column, and lane 0 stores the bf16 result.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from aiter.ops.flydsl.kernels.act import sigmoid_batch
from aiter.ops.flydsl.kernels.kernels_common import ceildiv

WARP_SIZE = 64

# One 128-bit load moves 8 bf16 values.
VEC = 8

# One wave-wide load covers 64 lanes * 8 values = 512 values of K.
CHUNK = WARP_SIZE * VEC

# The kernel keeps one piece of y per (token, chunk) in registers after the barrier,
# and its code size grows with the token count, so it is meant for decode only.
MAX_TOKENS = 8

# The output store moves cols_per_wave bf16 values at once, so its width depends on
# cols_per_wave.
_STORE_COPY = {
    1: fx.rocdl.BufferCopy16b,
    2: fx.rocdl.BufferCopy32b,
    4: fx.rocdl.BufferCopy64b,
    8: fx.rocdl.BufferCopy128b,
}


def check_gdn_gated_rmsnorm_out_proj_shape(
    m, k, n, head_dim, cols_per_wave, waves_per_block
):
    """Return None when the kernel supports this shape and tile, else the reason."""
    if not 1 <= m <= MAX_TOKENS:
        return f"token count {m} is outside 1 to {MAX_TOKENS}"
    lanes_per_head = head_dim // VEC
    if head_dim % VEC or lanes_per_head == 0 or WARP_SIZE % lanes_per_head:
        return f"head_dim {head_dim} does not split into 8-value lane groups of a wave"
    if CHUNK % head_dim:
        return f"head_dim {head_dim} does not divide the {CHUNK}-value chunk"
    if k <= 0 or n <= 0:
        return f"K={k} and N={n} must be positive"
    if k % CHUNK:
        return f"K={k} is not a multiple of {CHUNK}"
    if cols_per_wave not in _STORE_COPY:
        return f"cols_per_wave must be one of {sorted(_STORE_COPY)}"
    if waves_per_block not in (4, 8):
        return "waves_per_block must be 4 or 8"
    if n % (cols_per_wave * waves_per_block):
        return f"N={n} is not a multiple of cols_per_wave * waves_per_block"
    return None


def build_gdn_gated_rmsnorm_out_proj_module(
    m: int,
    k: int,
    n: int,
    head_dim: int,
    eps: float,
    cols_per_wave: int,
    waves_per_block: int,
):
    """Build a launcher for ``out[m, n] = RMSNormGated(x, z) @ W[n, k].T``.

    Launcher arguments, all bf16 and contiguous:

    * ``X`` and ``Z``: ``[m * k]``, the GDN core output and the gate of all tokens,
      flattened, so there are ``k = num_heads * head_dim`` values per token.
    * ``NormW``: ``[head_dim]``, the RMSNormGated weight that every head shares.
    * ``W``: ``[n, k]``, the out_proj weight in the ``nn.Linear`` layout.
    * ``Out``: ``[m, n]``, the result.
    * ``stream``: the stream to launch on.
    """
    reason = check_gdn_gated_rmsnorm_out_proj_shape(
        m, k, n, head_dim, cols_per_wave, waves_per_block
    )
    if reason is not None:
        raise ValueError(f"unsupported GDN gated RMSNorm + out_proj shape: {reason}")

    lanes_per_head = head_dim // VEC
    num_chunks = k // CHUNK
    # A piece is one (token, chunk) pair, 512 values of y. Piece p covers token
    # p // num_chunks and chunk p % num_chunks, so its 8-value vectors start at
    # vector index p * WARP_SIZE of the flattened [m * k] inputs.
    num_pieces = m * num_chunks
    prologue_rounds = ceildiv(num_pieces, waves_per_block)
    head_reduce_steps = int(math.log2(lanes_per_head))
    wave_reduce_steps = int(math.log2(WARP_SIZE))
    inv_head_dim = 1.0 / head_dim
    grid_x = n // (cols_per_wave * waves_per_block)
    block_x = WARP_SIZE * waves_per_block

    # The shape and the tile are in the kernel name, so that profiler output tells the
    # configurations apart.
    kernel_name = (
        f"gdn_gated_rmsnorm_out_proj_m{m}_k{k}_n{n}_d{head_dim}"
        f"_c{cols_per_wave}_w{waves_per_block}"
    )

    @fx.struct
    class SharedStorage:
        y: fx.Array[fx.BFloat16, m * k, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_x, 1, 1])
    def gdn_gated_rmsnorm_out_proj_kernel(
        X: fx.Tensor,
        Z: fx.Tensor,
        NormW: fx.Tensor,
        W: fx.Tensor,
        Out: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        fm_fast = arith.FastMathFlags.fast

        # The buffer descriptors keep FlyDSL's default size of 0xFFFFFFFF bytes, so the
        # hardware does not bound any access. Every index below stays inside its tensor
        # because m, k, n, and head_dim are constants of this build and the wrapper
        # checks every tensor shape against them.
        w_buf = fx.rocdl.make_buffer_tensor(W)
        out_buf = fx.rocdl.make_buffer_tensor(Out)
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
        store_atom = fx.make_copy_atom(_STORE_COPY[cols_per_wave](), 16)

        def vec_view(row):
            # Split a 1-D row into 8-value vectors, so that index i selects values
            # 8 * i to 8 * i + 7.
            return fx.logical_divide(row, fx.make_layout(VEC, 1))

        def load_bf16x8(view, idx):
            r = fx.make_rmem_tensor(VEC, fx.BFloat16)
            fx.copy(load_atom, fx.slice(view, (None, idx)), r)
            return fx.memref_load_vec(r)

        # Issue every W load of this wave first, so the HBM read of W starts as early
        # as possible. Loads return in order, so the x and z loads below also wait for
        # these W loads before this wave computes its part of the norm.
        col0 = (bid * waves_per_block + wave) * cols_per_wave
        w_raw = []
        for j in range_constexpr(cols_per_wave):
            w_row = vec_view(fx.slice(w_buf, (col0 + j, None)))
            w_raw.append(
                [
                    load_bf16x8(w_row, lane + c * WARP_SIZE)
                    for c in range_constexpr(num_chunks)
                ]
            )

        x_vecs = vec_view(fx.rocdl.make_buffer_tensor(X))
        z_vecs = vec_view(fx.rocdl.make_buffer_tensor(Z))
        # Every chunk puts lane l at the same offset inside a head,
        # 8 * (l % lanes_per_head), so one load of the norm weight serves all pieces.
        nw = load_bf16x8(
            vec_view(fx.rocdl.make_buffer_tensor(NormW)), lane % lanes_per_head
        ).to(fx.Float32)

        def y_piece(piece):
            """Return this lane's 8 values of y for one piece, rounded to bf16."""
            idx = piece * WARP_SIZE + lane
            xv = load_bf16x8(x_vecs, idx).to(fx.Float32)
            zv = load_bf16x8(z_vecs, idx).to(fx.Float32)

            # Sum the squares of one head. The lanes of a head are consecutive, so XOR
            # offsets below lanes_per_head never leave the head.
            sumsq = (xv * xv).reduce(ReductionOp.ADD, fastmath=fm_fast)
            for s in range_constexpr(head_reduce_steps):
                offset = lanes_per_head // (2 << s)
                sumsq = sumsq.addf(
                    sumsq.shuffle_xor(offset, WARP_SIZE), fastmath=fm_fast
                )
            rrms = fmath.rsqrt(sumsq * inv_head_dim + eps, fastmath=fm_fast)

            zs = [zv[i] for i in range_constexpr(VEC)]
            sig = sigmoid_batch(zs)
            gate = fx.Vector.from_elements(
                [zs[i] * sig[i] for i in range_constexpr(VEC)], fx.Float32
            )

            # The unfused path rounds y to bf16 before the GEMM reads it. Rounding here
            # too keeps this kernel equal to it up to the GEMM summation order.
            return ((xv * rrms) * nw * gate).to(fx.BFloat16)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_vecs = vec_view(lds.y.view(fx.make_layout(m * k, 1)))
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        for rnd in range_constexpr(prologue_rounds):
            piece = wave + rnd * waves_per_block
            if piece < num_pieces:
                r_piece = fx.make_rmem_tensor(VEC, fx.BFloat16)
                fx.memref_store_vec(y_piece(piece), r_piece)
                fx.copy(
                    lds_atom,
                    r_piece,
                    fx.slice(lds_vecs, (None, piece * WARP_SIZE + lane)),
                )
        gpu.barrier()

        y = []
        for t in range_constexpr(m):
            token_chunks = []
            for c in range_constexpr(num_chunks):
                r_lds = fx.make_rmem_tensor(VEC, fx.BFloat16)
                fx.copy(
                    lds_atom,
                    fx.slice(lds_vecs, (None, (t * num_chunks + c) * WARP_SIZE + lane)),
                    r_lds,
                )
                token_chunks.append(fx.memref_load_vec(r_lds))
            y.append(token_chunks)

        def bf16_pair(v, p):
            # Values 2 * p and 2 * p + 1 of an 8-value bf16 vector share one 32-bit
            # register, so this selects a register and emits no instruction.
            return v.shuffle(v, [2 * p, 2 * p + 1])

        def dot2(a, b, acc):
            # acc + a[0] * b[0] + a[1] * b[1] as one v_dot2_f32_bf16. FlyDSL has no
            # fx-level wrapper for this instruction, so the ROCDL op is called directly.
            return fx.Float32(
                fx.rocdl.fdot2_f32_bf16_(
                    T.f32, a.ir_value(), b.ir_value(), acc.ir_value()
                )
            )

        # Dot the W rows of this wave with y. Two accumulators per (column, token)
        # halve the chain of dependent dot2 instructions, so consecutive ones overlap.
        partial = []
        for j in range_constexpr(cols_per_wave):
            col_partial = []
            for t in range_constexpr(m):
                accs = [fx.Float32(0.0), fx.Float32(0.0)]
                for c in range_constexpr(num_chunks):
                    for p in range_constexpr(VEC // 2):
                        accs[p % 2] = dot2(
                            bf16_pair(w_raw[j][c], p),
                            bf16_pair(y[t][c], p),
                            accs[p % 2],
                        )
                col_partial.append(accs[0] + accs[1])
            partial.append(col_partial)

        # Sum each lane's partial dot products over the wave. After the XOR butterfly
        # every lane holds the full sum, and lane 0 writes it.
        totals = []
        for j in range_constexpr(cols_per_wave):
            col_totals = []
            for t in range_constexpr(m):
                s_val = partial[j][t]
                for r in range_constexpr(wave_reduce_steps):
                    offset = WARP_SIZE // (2 << r)
                    s_val = s_val.addf(
                        s_val.shuffle_xor(offset, WARP_SIZE), fastmath=fm_fast
                    )
                col_totals.append(s_val)
            totals.append(col_totals)

        if lane == 0:
            for t in range_constexpr(m):
                out_row = fx.logical_divide(
                    fx.slice(out_buf, (t, None)), fx.make_layout(cols_per_wave, 1)
                )
                vals = fx.Vector.from_elements(
                    [totals[j][t] for j in range_constexpr(cols_per_wave)],
                    fx.Float32,
                ).to(fx.BFloat16)
                r_out = fx.make_rmem_tensor(cols_per_wave, fx.BFloat16)
                fx.memref_store_vec(vals, r_out)
                fx.copy(
                    store_atom,
                    r_out,
                    fx.slice(out_row, (None, bid * waves_per_block + wave)),
                )

    @flyc.jit
    def launch_gdn_gated_rmsnorm_out_proj(
        X: fx.Tensor,
        Z: fx.Tensor,
        NormW: fx.Tensor,
        W: fx.Tensor,
        Out: fx.Tensor,
        stream: fx.Stream,
    ):
        gdn_gated_rmsnorm_out_proj_kernel(X, Z, NormW, W, Out).launch(
            grid=(grid_x, 1, 1),
            block=(block_x, 1, 1),
            stream=stream,
        )

    return launch_gdn_gated_rmsnorm_out_proj

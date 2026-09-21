# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated Delta Net decode recurrence for packed (varlen) batches.

Companion to ``gdr_decode.py``. That kernel serves batch-major decode
(``[B, S, H, D]`` with ``S`` baked in, one state read and one state write). This
one serves the layout an SGLang-style server actually hands over during
speculative decoding:

* **packed / varlen** -- ``q``/``k``/``v`` arrive as ``[1, T, H, D]`` with a
  ``cu_seqlens`` marking sequence starts, not as a batch dimension;
* **EAGLE target-verify** -- the draft tokens of one step are verified together,
  so the caller wants each draft step's post-update state snapshotted into an
  ``intermediate_states`` buffer and the final write-back *suppressed* (the
  accepted prefix is committed separately).

Neither is expressible through ``flydsl_gdr_decode``, which is why this is a
separate entry point rather than more flags on that one.

Decomposition
-------------
K is split across lanes rather than living inside one lane::

    lane owns  v = vb + tid // KSPLIT,  k in [ (tid % KSPLIT) * KLOCAL, +KLOCAL )

so ``KSPLIT`` lanes cooperate on one v column and every reduction over K -- the
two l2-norms, the delta-rule projection and the output projection -- becomes
``log2(KSPLIT)`` ``shuffle_xor`` steps. No LDS and no barriers. Each lane
carries only ``KLOCAL`` floats of state, and the ``KSPLIT`` lanes of a v group
cover one contiguous ``K * 2``-byte state row, so the loads coalesce.

Two constraints that are easy to reintroduce by accident
--------------------------------------------------------
1. The token loop is unrolled at compile time (``TSTATIC``), not a runtime
   ``range``. A global store inside a FlyDSL runtime carry loop miscompiles;
   the draft-token count is a small constant anyway, so this costs nothing.
2. Only one lane per v group writes the output, but wrapping a store in an
   ``if`` triggers (1). Non-writing lanes are instead steered past the output
   descriptor's ``num_records`` so the hardware drops them. That *requires*
   ``max_size=False`` with an explicit ``num_records_bytes``; with the default
   unbounded descriptor the steered stores are not dropped and scribble far
   past the tensor.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

KDIM = 128
HALF = 8  # b128 is the widest buffer load: 8 bf16


def _dt(name: str):
    return {"torch.bfloat16": fx.BFloat16, "torch.float32": fx.Float32}[name]


@flyc.jit
def _load_vec(buf, offset, width: fx.Constexpr[int], dtype: fx.Constexpr):
    view = fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(width, 1))
    reg = fx.make_rmem_tensor(width, dtype)
    fx.copy_atom_call(
        fx.make_copy_atom(fx.rocdl.BufferCopy(width * dtype.width), dtype), view, reg
    )
    return reg.load()


@flyc.jit
def _store_vec(buf, offset, value, dtype: fx.Constexpr):
    reg = fx.make_rmem_tensor(value.numel, dtype)
    reg.store(value)
    view = fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(value.numel, 1))
    fx.copy_atom_call(
        fx.make_copy_atom(fx.rocdl.BufferCopy(value.numel * dtype.width), dtype),
        reg,
        view,
    )


@functools.lru_cache(maxsize=1024)
def create_gdn_decode_verify_kernel(
    num_v_heads: int,
    num_k_heads: int,
    head_v_dim: int,
    tokens_per_seq: int,
    out_bytes: int,
    state_slot_stride: int,
    q_token_stride: int,
    k_token_stride: int,
    v_token_stride: int,
    a_token_stride: int,
    b_token_stride: int,
    scale: float,
    softplus_beta: float,
    softplus_threshold: float,
    update_state: bool,
    cache_states: bool,
    cache_steps: int,
    a_dtype: str,
    b_dtype: str,
    KSPLIT: int = 8,
    BLOCK: int = 256,
):
    HV, Hg, V, TSTATIC = num_v_heads, num_k_heads, head_v_dim, tokens_per_seq
    VTILE = BLOCK // KSPLIT
    KLOCAL = KDIM // KSPLIT
    NHALF = KLOCAL // HALF
    if KLOCAL % HALF or VTILE * KSPLIT != BLOCK or V % VTILE:
        raise ValueError(
            f"bad geometry: KSPLIT={KSPLIT} BLOCK={BLOCK} V={V} -> "
            f"KLOCAL={KLOCAL} VTILE={VTILE}"
        )
    group = HV // Hg
    A_DT, B_DT = _dt(a_dtype), _dt(b_dtype)
    OOB = out_bytes  # bytes >= elements for bf16, so always past the descriptor

    @flyc.jit
    def _group_sum(x):
        # Unrolled explicitly: the AST rewriter rebinds names captured inside
        # range_constexpr bodies, which would break an accumulator.
        s = x
        if const_expr(KSPLIT > 1):
            s = s + fx.gpu.shuffle_xor(s, 1, 64)
        if const_expr(KSPLIT > 2):
            s = s + fx.gpu.shuffle_xor(s, 2, 64)
        if const_expr(KSPLIT > 4):
            s = s + fx.gpu.shuffle_xor(s, 4, 64)
        if const_expr(KSPLIT > 8):
            s = s + fx.gpu.shuffle_xor(s, 8, 64)
        if const_expr(KSPLIT > 16):
            s = s + fx.gpu.shuffle_xor(s, 16, 64)
        if const_expr(KSPLIT > 32):
            s = s + fx.gpu.shuffle_xor(s, 32, 64)
        return s

    @flyc.kernel
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        cu: fx.Tensor,
        icache: fx.Tensor,
        icache_idx: fx.Tensor,
    ):
        tid = fx.Int64(fx.thread_idx.x)
        hv = fx.Int64(fx.block_idx.x)
        vb = fx.Int64(fx.block_idx.y) * VTILE
        seq = fx.Int64(fx.block_idx.z)

        kpart = tid % KSPLIT
        vcol = vb + tid // KSPLIT
        kbase = kpart * KLOCAL
        ih = hv // group

        bos = fx.Int64(cu[seq])
        slot = fx.Int64(indices[seq])
        cslot = fx.Int64(0)
        if const_expr(cache_states):
            cslot = fx.Int64(icache_idx[seq])

        qb = fx.rocdl.make_buffer_tensor(q)
        kb = fx.rocdl.make_buffer_tensor(k)
        vbuf = fx.rocdl.make_buffer_tensor(v)
        ab = fx.rocdl.make_buffer_tensor(a)
        bb = fx.rocdl.make_buffer_tensor(b)
        # Bounded on purpose -- the predicated output store relies on
        # out-of-range offsets being dropped. See the module docstring.
        ob = fx.rocdl.make_buffer_tensor(
            out, max_size=False, num_records_bytes=fx.Int64(out_bytes)
        )
        # Rebase before building the 32-bit-offset resource: a whole state pool
        # can exceed the 32-bit offset range.
        state_slot = fx.make_view(
            fx.get_iter(state) + slot * state_slot_stride,
            fx.make_layout(HV * V * KDIM, 1),
        )
        ss = fx.rocdl.make_buffer_tensor(state_slot)
        if const_expr(cache_states):
            cache_slot = fx.make_view(
                fx.get_iter(icache) + cslot * (cache_steps * HV * V * KDIM),
                fx.make_layout(cache_steps * HV * V * KDIM, 1),
            )
            cb = fx.rocdl.make_buffer_tensor(cache_slot)

        alog = fx.Float32(A_log[hv])
        dtb = fx.Float32(dt_bias[hv])
        neg_exp_alog = -fx.exp(alog)

        sbase = hv * V * KDIM + vcol * KDIM + kbase
        h = [fx.Vector.filled(HALF, 0.0, fx.Float32) for _ in range(NHALF)]
        if slot >= 0:
            h = [
                _load_vec(ss, sbase + j * HALF, HALF, fx.BFloat16).to(fx.Float32)
                for j in range(NHALF)
            ]

        for t in range_constexpr(TSTATIC):
            tok = bos + t
            koff = tok * k_token_stride + ih * KDIM + kbase
            qoff = tok * q_token_stride + ih * KDIM + kbase
            kvec = [
                _load_vec(kb, koff + j * HALF, HALF, fx.BFloat16).to(fx.Float32)
                for j in range(NHALF)
            ]
            qvec = [
                _load_vec(qb, qoff + j * HALF, HALF, fx.BFloat16).to(fx.Float32)
                for j in range(NHALF)
            ]

            # Reductions built as expressions: a statement-level `for` in a
            # kernel body is rewritten into a runtime scf.for, which would make
            # the index a DSL value and break the Python-list subscript.
            sk_local = functools.reduce(lambda p, c: p + c, [x * x for x in kvec])
            sq_local = functools.reduce(lambda p, c: p + c, [x * x for x in qvec])
            sk = _group_sum(sk_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            sq = _group_sum(sq_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            rk = fx.Float32(1.0) / fx.sqrt(sk + fx.Float32(1e-6))
            rq = scale / fx.sqrt(sq + fx.Float32(1e-6))
            kvec = [x * rk for x in kvec]
            qvec = [x * rq for x in qvec]

            av = fx.Float32(
                _load_vec(ab, tok * a_token_stride + hv, 1, A_DT)[0].to(fx.Float32)
            )
            bv_in = fx.Float32(
                _load_vec(bb, tok * b_token_stride + hv, 1, B_DT)[0].to(fx.Float32)
            )
            vval = fx.Float32(
                _load_vec(vbuf, tok * v_token_stride + hv * V + vcol, 1, fx.BFloat16)[
                    0
                ].to(fx.Float32)
            )

            x = av + dtb
            bx = softplus_beta * x
            sp = (bx <= softplus_threshold).select(
                (fx.Float32(1.0) / softplus_beta)
                * fx.log(fx.Float32(1.0) + fx.exp(bx)),
                x,
            )
            decay = fx.exp(neg_exp_alog * sp)
            beta = fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-bv_in))

            hd = [y * decay for y in h]
            dv_local = functools.reduce(
                lambda p, c: p + c, [hd[j] * kvec[j] for j in range(NHALF)]
            )
            dv = _group_sum(dv_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            vnew = (vval - dv) * beta
            h = [hd[j] + kvec[j] * vnew for j in range(NHALF)]

            ov_local = functools.reduce(
                lambda p, c: p + c, [h[j] * qvec[j] for j in range(NHALF)]
            )
            ov = _group_sum(ov_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            oaddr = (kpart == 0).select((tok * HV + hv) * V + vcol, fx.Int64(OOB))
            _store_vec(
                ob,
                oaddr,
                fx.Vector.from_elements([ov.to(fx.BFloat16)], fx.BFloat16),
                fx.BFloat16,
            )

            if const_expr(cache_states):
                if cslot >= 0:
                    for j in range_constexpr(NHALF):
                        _store_vec(
                            cb,
                            t * (HV * V * KDIM) + sbase + j * HALF,
                            h[j].to(fx.BFloat16),
                            fx.BFloat16,
                        )

        if const_expr(update_state):
            if slot >= 0:
                for j in range_constexpr(NHALF):
                    _store_vec(
                        ss, sbase + j * HALF, h[j].to(fx.BFloat16), fx.BFloat16
                    )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        cu: fx.Tensor,
        icache: fx.Tensor,
        icache_idx: fx.Tensor,
        n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
            q, k, v, a, b, A_log, dt_bias, state, indices, out, cu, icache, icache_idx
        ).launch(grid=(HV, V // VTILE, n), block=(BLOCK,), stream=stream)

    return launch

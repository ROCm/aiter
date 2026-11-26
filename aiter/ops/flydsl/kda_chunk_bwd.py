# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL KDA (``chunk_kimi_delta_attn``) chunk backward.

``chunk_kda_bwd_flydsl_full`` is the FlyDSL counterpart of the Triton
``chunk_delta_attn_bwd`` and returns the same 8-tuple. The forward's
intermediates are recomputed with the Triton forward prologue
(``chunk_delta_attn_fwd_prepare``); every backward stage is FlyDSL:

1. chunk-state recurrence ``h`` / ``v_new`` -- the GDN K5 kernel
   (``chunk_gated_delta_h``, per-channel ``USE_GK``), read in the layouts it
   writes (``h`` ``[B, NT, H, V, K]``, ``v_new`` ``[B, H, T, V]``), no copies;
2. ``dav``, ``dhu``, ``wy_dqkg``, ``intra``, ``gate_cumsum`` --
   ``kernels/kda_chunk_bwd.py``. ``intra`` also takes dq/dk back through the
   in-kernel l2norm and dbeta through the beta sigmoid; ``gate_cumsum`` takes
   dg through the gate activation and leaves the ``dA_log`` / ``dt_bias``
   partials.

Inputs and gradients are token-major ``[B, T, H, *]``, the layout of fla's
``chunk_kda`` and of the projections around it, so a fused layer backward
(#8739) and the conv backward (#8726) consume dq/dk/dv/dg/dbeta in place.

``chunk_kda_bwd_flydsl`` adapts it to #8739's ``register_chunk_bwd`` 7-tuple
contract; ``CHUNK_BWD_FUSES`` names the prologues it handles in kernel.
"""

from __future__ import annotations

import os

import torch

from .kernels.kda_chunk_bwd import (
    BT,
    compile_dav,
    compile_dhu,
    compile_gate_cumsum,
    compile_intra,
    compile_wy_dqkg,
)
from .kernels.tensor_shim import _run_compiled
from .linear_attention_prefill_kernels import (
    _get_or_compile_opt,
    _hipeq_select_bv,
    _is_gfx942,
    _placeholder_pair,
)

CHUNK_BWD_FUSES = frozenset({"qk_l2norm", "gate", "beta_sigmoid"})

# AITER_KDA_FLYDSL_CHECK=1 synchronizes after each stage and raises on the first
# non-finite output, naming the stage. A debugging aid, off by default.
_CHECK = os.environ.get("AITER_KDA_FLYDSL_CHECK", "0") == "1"


def _check(stage, **tensors):
    if not _CHECK:
        return
    for name, t in tensors.items():
        if t is not None and not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).nonzero()
            raise FloatingPointError(
                f"KDA FlyDSL backward: {stage} wrote non-finite {name} "
                f"{tuple(t.shape)}: {bad.shape[0]} entries, first at {bad[0].tolist()}"
            )


def kda_bwd_flydsl_supported(
    *, H: int, HV: int, K: int, V: int, chunk_size: int, is_varlen: bool, dtype
) -> bool:
    """Calls the FlyDSL backward takes; everything else stays on Triton."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    # The stage kernels stage up to ~147 KB of LDS per workgroup (gfx950 has
    # 160 KB, gfx942 64 KB); the K5 recurrence bakes BT = 64 and K = 128.
    return (
        get_gfx_runtime() == "gfx950"
        and K == 128
        and V == 128
        and chunk_size == BT
        and H == HV
        and not is_varlen
        and dtype == torch.bfloat16
    )


def kda_bwd_fwd_h_flydsl(kg, w, u, g_cumsum, initial_state, state_v_first):
    """Per-chunk states and ``v_new = u - w @ h`` on the K5 kernel.

    kg, w, u: token-major ``[B, T, H, *]`` bf16 (kg decayed to its chunk end);
    g_cumsum: ``[B, T, H, K]`` log2 chunk cumsum. Returns ``h`` as
    ``[B, NT, H, V, K]`` and ``v_new`` as ``[B, H, T, V]``.
    """
    B, T, H, K = kg.shape
    V = u.shape[-1]
    NT = (T + BT - 1) // BT
    gk = g_cumsum.to(torch.float32).contiguous()
    h0 = None
    if initial_state is not None:
        h0 = initial_state.to(torch.float32)
        if not state_v_first:
            h0 = h0.transpose(-1, -2)
        h0 = h0.contiguous()
    BV = _hipeq_select_bv(kg.device, H, B * NT, NT)
    launch_fn = _get_or_compile_opt(
        K,
        V,
        BT,
        BV,
        H,
        H,
        False,  # use_g: KDA's decay is per channel only
        True,  # use_gk
        h0 is not None,
        False,  # store_fs
        True,  # save_vn
        False,  # is_varlen
        False,  # wu_contig: w/u are token-major
        state_bf16=False,
        g_log2_scaled=True,
        sched_gfx942=_is_gfx942(),
        bf16_convert_trunc=False,
        snapshot_bf16=True,
    )
    dummy, int32_dummy = _placeholder_pair(kg.device)
    h = kg.new_empty(B, NT, H, V, K)
    v_new = kg.new_empty(B, H, T, V)
    _run_compiled(
        launch_fn,
        kg.contiguous(),
        u.contiguous(),
        w.contiguous(),
        v_new,
        dummy,
        gk,
        h,
        h0 if h0 is not None else dummy,
        dummy,
        int32_dummy,
        int32_dummy,
        int32_dummy,
        T,
        T,
        B,
        V // BV,
        B * H,
        torch.cuda.current_stream(),
    )
    return h, v_new


def chunk_kda_bwd_flydsl_full(
    do,
    dht,
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    chunk_size=BT,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    state_v_first=False,
):
    """FlyDSL ``chunk_delta_attn_bwd``: ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0)``.

    Same arguments as the Triton ``chunk_delta_attn_bwd`` without varlen;
    the caller checks ``kda_bwd_flydsl_supported`` first.
    """
    from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_fwd import (
        chunk_delta_attn_fwd_prepare,
    )

    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = (T + BT - 1) // BT
    dev = q.device
    stream = torch.cuda.current_stream()
    dummy, _ = _placeholder_pair(dev)
    f32 = {"device": dev, "dtype": torch.float32}

    qn, kn, bn, gc, w, u, qg, kg, Aqk, Akk = chunk_delta_attn_fwd_prepare(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=BT,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        disable_recompute=True,
    )
    _check("fwd_prepare (Triton)", q=qn, k=kn, g=gc, w=w, u=u, qg=qg, kg=kg, Akk=Akk)
    h, v_new = kda_bwd_fwd_h_flydsl(kg, w, u, gc, initial_state, state_v_first)
    _check("fwd_h", h=h, v_new=v_new)
    do = do.contiguous()

    dAqk = torch.empty(B, T, H, BT, **f32)
    dv = torch.empty_like(v)
    _run_compiled(
        compile_dav(H, v.dtype, Aqk.dtype),
        v_new,
        Aqk,
        do,
        dAqk,
        dv,
        T,
        float(scale),
        NT,
        B * H,
        stream,
    )
    _check("dav", dAqk=dAqk, dv=dv)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = (
        torch.empty_like(initial_state, dtype=torch.float32)
        if initial_state is not None
        else None
    )
    dv2 = torch.empty_like(dv)
    _run_compiled(
        compile_dhu(
            H,
            kg.dtype,
            v.dtype,
            initial_state is not None,
            dht is not None,
            state_v_first,
        ),
        qg,
        kg,
        w,
        gc,
        do,
        dv,
        dht.contiguous() if dht is not None else dummy,
        dh,
        dh0 if dh0 is not None else dummy,
        dv2,
        T,
        float(scale),
        B * H,
        stream,
    )
    _check("dhu", dh=dh, dh0=dh0, dv2=dv2)

    dq = torch.empty(B, T, H, K, **f32)
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    dv = torch.empty_like(v)
    db = torch.empty(B, T, H, **f32)
    dAkk = torch.empty_like(dAqk)
    _run_compiled(
        compile_wy_dqkg(H, qn.dtype, v.dtype, bn.dtype, Akk.dtype, h.dtype),
        qn,
        kn,
        v,
        v_new,
        gc,
        bn,
        Akk,
        h,
        do,
        dh,
        dv2,
        dq,
        dk,
        dv,
        dg,
        db,
        dAkk,
        T,
        float(scale),
        NT,
        B * H,
        stream,
    )
    _check("wy_dqkg", dq=dq, dk=dk, dv=dv, dg=dg, db=db, dAkk=dAkk)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    dg2 = torch.empty(B, T, H, K, **f32)
    db2 = torch.empty_like(beta)
    _run_compiled(
        compile_intra(
            H,
            qn.dtype,
            bn.dtype,
            q.dtype,
            beta.dtype,
            q.dtype,
            beta.dtype,
            use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel,
        ),
        qn,
        kn,
        gc,
        bn,
        dAqk,
        dAkk,
        dq,
        dk,
        dg,
        db,
        q,
        k,
        beta,
        dq2,
        dk2,
        dg2,
        db2,
        T,
        NT,
        B * H,
        stream,
    )
    _check("intra", dq=dq2, dk=dk2, dg=dg2, db=db2)

    use_bias = use_gate_in_kernel and dt_bias is not None
    dA_part = torch.empty(NT, B, H, K, **f32) if use_gate_in_kernel else None
    db_part = torch.empty(NT, B, H, K, **f32) if use_bias else None
    dg_out = torch.empty_like(g)
    _run_compiled(
        compile_gate_cumsum(
            H, g.dtype, g.dtype, use_gate_in_kernel, use_bias, lower_bound is not None
        ),
        dg2,
        g,
        A_log.float().contiguous() if use_gate_in_kernel else dummy,
        dt_bias.float().contiguous() if use_bias else dummy,
        dg_out,
        dA_part if dA_part is not None else dummy,
        db_part if db_part is not None else dummy,
        T,
        0.0 if lower_bound is None else float(lower_bound),
        NT,
        B * H,
        stream,
    )
    _check("gate_cumsum", dg=dg_out, dA_part=dA_part, db_part=db_part)
    dA_log = dA_part.sum((0, 1, 3)).to(A_log.dtype) if use_gate_in_kernel else None
    ddt_bias = db_part.sum((0, 1)).reshape(-1).to(dt_bias.dtype) if use_bias else None
    return (
        dq2,
        dk2,
        dv,
        dg_out,
        db2,
        dA_log,
        ddt_bias,
        dh0.to(initial_state.dtype) if dh0 is not None else None,
    )


def chunk_kda_bwd_flydsl(
    do,
    q,
    k,
    v,
    g,
    beta,
    scale,
    chunk_size=BT,
    *,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    use_beta_sigmoid_in_kernel=False,
):
    """#8739 ``register_chunk_bwd`` contract: ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias)``.

    q, k, g ``[B, T, H, K]``, v and do ``[B, T, H, V]``, beta ``[B, T, H]``,
    no initial or final state. q/k/v/do are computed in bf16; each gradient
    comes back in its input's dtype. Falls back to the Triton backward when
    the shape is outside ``kda_bwd_flydsl_supported``.
    """
    in_dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype)
    bf = torch.bfloat16
    q, k, v, do = (t.to(bf).contiguous() for t in (q, k, v, do))
    g, beta = g.contiguous(), beta.contiguous()
    H, K = q.shape[2:]
    kw = {
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "use_gate_in_kernel": use_gate_in_kernel,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
    }
    if kda_bwd_flydsl_supported(
        H=H,
        HV=v.shape[2],
        K=K,
        V=v.shape[-1],
        chunk_size=chunk_size,
        is_varlen=False,
        dtype=q.dtype,
    ):
        out = chunk_kda_bwd_flydsl_full(
            do, None, q, k, v, g, beta, scale, chunk_size=chunk_size, **kw
        )
    else:
        from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_bwd import (
            chunk_delta_attn_bwd,
        )

        out = chunk_delta_attn_bwd(
            do, None, q, k, v, g, beta, scale, chunk_size=chunk_size, **kw
        )
    dq, dk, dv, dg, db, dA_log, ddt = out[:7]
    dq, dk, dv, dg, db = (t.to(dt) for t, dt in zip((dq, dk, dv, dg, db), in_dtypes))
    return dq, dk, dv, dg, db, dA_log, ddt


def register_with_kda_layer() -> bool:
    """Install ``chunk_kda_bwd_flydsl`` as #8739's chunk backward, if present."""
    try:
        from aiter.ops.kda_fused_layer import register_chunk_bwd
    except ImportError:
        return False
    register_chunk_bwd(chunk_kda_bwd_flydsl, fuses=CHUNK_BWD_FUSES)
    return True

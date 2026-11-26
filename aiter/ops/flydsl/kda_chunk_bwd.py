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

import csv
import os

import torch

from .kernels.kda_chunk_bwd import (
    BT,
    compile_dav,
    compile_dav_mfma,
    compile_dhu,
    compile_dhu_mfma,
    compile_gate_cumsum,
    compile_intra,
    compile_intra_mfma,
    compile_wy_dqkg,
    compile_wy_dqkg_mfma,
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
# AITER_KDA_FLYDSL_MFMA=0 selects the fp32-FMA stage kernels instead of their
# MFMA versions (A/B and bisection aid).
_MFMA = os.environ.get("AITER_KDA_FLYDSL_MFMA", "1") == "1"
# AITER_KDA_FLYDSL_PROFILE=1 times every stage with CUDA events and logs it.
_PROFILE = os.environ.get("AITER_KDA_FLYDSL_PROFILE", "0") == "1"


# Forward-intermediate recompute ("prepare") backends: name -> fn with the
# signature and 10-tuple result of the Triton ``chunk_delta_attn_fwd_prepare``.
_PREPARE_IMPLS = {}


def register_fwd_prepare(name: str, fn) -> None:
    """Register a forward-prepare backend for the FlyDSL backward.

    ``fn(*, q, k, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size,
    safe_gate, lower_bound, use_gate_in_kernel, A_log, dt_bias,
    use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute)``
    returns ``(q, k, beta, g_cumsum, w, u, qg, kg, Aqk, Akk)`` in the Triton
    prepare's layouts and dtypes. It is used only when selected, through
    ``prepare=name`` or ``AITER_KDA_BWD_PREPARE=name``; the default stays
    ``"triton"``.
    """
    if name == "triton":
        raise ValueError('"triton" is the built-in default and cannot be replaced.')
    _PREPARE_IMPLS[name] = fn


# Tuned configs (AITER_CONFIG_KDA_CHUNK_BWD): key (gfx, cu_num, N, T, H, varlen),
# N sequences of T tokens (varlen: T = total tokens // N).
DEFAULT_DHU_BV = 32
_TUNED = None


def _tuned_table():
    global _TUNED
    if _TUNED is None:
        table = {}
        try:
            from aiter.jit.core import AITER_CONFIGS

            with open(
                AITER_CONFIGS.AITER_CONFIG_KDA_CHUNK_BWD_FILE, encoding="utf-8"
            ) as f:
                for row in csv.DictReader(f):
                    key = (
                        row["gfx"].strip(),
                        int(row["cu_num"]),
                        int(row["N"]),
                        int(row["T"]),
                        int(row["H"]),
                        int(row["varlen"]),
                    )
                    table[key] = {
                        "dhu_bv": int(row["dhu_bv"]),
                        "fwd_h_bv": int(row["fwd_h_bv"] or 0) or None,
                        "prepare": (row.get("prepare") or "").strip() or "triton",
                    }
        except (OSError, KeyError, ValueError, ImportError) as exc:
            import logging

            logging.getLogger("aiter").warning(
                "KDA chunk bwd tuned configs unavailable (%s); using defaults", exc
            )
        _TUNED = table
    return _TUNED


def tuned_config(N: int, T: int, H: int, varlen: bool) -> dict:
    """The tuned row for this shape on this GPU, or ``{}`` (defaults apply)."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    key = (get_gfx_runtime(), get_cu_num(), N, T, H, int(varlen))
    return _tuned_table().get(key, {})


def _fwd_prepare_fn(prepare, tuned=None):
    name = prepare or os.environ.get("AITER_KDA_BWD_PREPARE")
    if not name:
        # A tuned prepare applies only if its backend is registered here.
        name = tuned if tuned in _PREPARE_IMPLS else "triton"
    if name == "triton":
        from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_fwd import (
            chunk_delta_attn_fwd_prepare,
        )

        return chunk_delta_attn_fwd_prepare
    if name not in _PREPARE_IMPLS:
        raise ValueError(
            f"KDA backward prepare {name!r} is not registered "
            f"(registered: {sorted(_PREPARE_IMPLS)} and 'triton')."
        )
    return _PREPARE_IMPLS[name]


class _Timer:
    def __init__(self):
        self.marks = []
        self.mark("start")

    def mark(self, name):
        if _PROFILE:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.marks.append((name, ev))

    def report(self, shape):
        if not _PROFILE:
            return
        torch.cuda.synchronize()
        parts = [
            f"{n}={a.elapsed_time(b) * 1e3:.0f}us"
            for (_, a), (n, b) in zip(self.marks, self.marks[1:])
        ]
        total = self.marks[0][1].elapsed_time(self.marks[-1][1]) * 1e3
        print(f"[kda flydsl bwd {shape}] {' '.join(parts)} total={total:.0f}us")


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
    *,
    H: int,
    HV: int,
    K: int,
    V: int,
    chunk_size: int,
    is_varlen: bool,
    dtype,
    B: int = 1,
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
        and (B == 1 or not is_varlen)
        and dtype == torch.bfloat16
    )


class _Varlen:
    """Chunk schedule of a varlen call: int32 ``cu_seqlens``, ``chunk_indices``
    (``[NT, 2]`` ``(sequence, chunk)``) and ``chunk_offsets`` (``[N + 1]``)."""

    def __init__(self, cu_seqlens, chunk_indices):
        from aiter.ops.triton._triton_kernels.gated_delta_net.utils.index import (
            prepare_chunk_offsets,
        )
        from aiter.ops.triton._triton_kernels.kimi_delta_attn.utils.index import (
            prepare_chunk_indices,
        )

        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        self.chunk_indices = chunk_indices
        self.cu = cu_seqlens.to(torch.int32).contiguous()
        self.ci = chunk_indices.to(torch.int32).contiguous()
        self.co = prepare_chunk_offsets(cu_seqlens, BT).to(torch.int32).contiguous()
        self.N = cu_seqlens.numel() - 1
        self.NT = chunk_indices.shape[0]


def kda_bwd_fwd_h_flydsl(
    kg, w, u, g_cumsum, initial_state, state_v_first, cu_seqlens=None, bv=None
):
    """Per-chunk states and ``v_new = u - w @ h`` on the K5 kernel.

    kg, w, u: token-major ``[B, T, H, *]`` bf16 (kg decayed to its chunk end);
    g_cumsum: ``[B, T, H, K]`` log2 chunk cumsum. Returns ``h`` as
    ``[B, NT, H, V, K]`` and ``v_new`` as ``[B, H, T, V]``. With
    ``cu_seqlens`` (``B = 1``) each sequence starts from its own state and
    ``NT`` counts the chunks of all sequences.
    """
    from .linear_attention_prefill_kernels import (
        _hipeq_varlen_host_metadata,
        _resolve_prologue,
    )

    B, T, H, K = kg.shape
    V = u.shape[-1]
    is_varlen = cu_seqlens is not None
    dummy, int32_dummy = _placeholder_pair(kg.device)
    if is_varlen:
        NT, chunk_offsets, kernel_cu, N, _ = _resolve_prologue(cu_seqlens, BT, 0, 0, T)
        total_chunks, max_seq_chunks = _hipeq_varlen_host_metadata(chunk_offsets)
        cu_arg = kernel_cu.to(torch.int32).contiguous()
        co_arg = chunk_offsets.to(torch.int32).contiguous()
    else:
        NT = (T + BT - 1) // BT
        N, total_chunks, max_seq_chunks = B, B * NT, NT
        cu_arg = co_arg = int32_dummy
    gk = g_cumsum.to(torch.float32).contiguous()
    h0 = None
    if initial_state is not None:
        h0 = initial_state.to(torch.float32)
        if not state_v_first:
            h0 = h0.transpose(-1, -2)
        h0 = h0.contiguous()
    BV = bv or _hipeq_select_bv(kg.device, H, total_chunks, max_seq_chunks)
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
        is_varlen,
        False,  # wu_contig: w/u are token-major
        state_bf16=False,
        g_log2_scaled=True,
        sched_gfx942=_is_gfx942(),
        bf16_convert_trunc=False,
        snapshot_bf16=True,
    )
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
        cu_arg,
        co_arg,
        int32_dummy,
        T,
        T,
        N,
        V // BV,
        N * H,
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
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=BT,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    state_v_first=False,
    prepare=None,
    dhu_bv=None,
    fwd_h_bv=None,
):
    """FlyDSL ``chunk_delta_attn_bwd``: ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0)``.

    Same arguments as the Triton ``chunk_delta_attn_bwd``; the caller checks
    ``kda_bwd_flydsl_supported`` first. With ``cu_seqlens`` (``B = 1``), chunks
    stop at sequence edges, every sequence has its own state (``initial_state``
    / ``dht`` / ``dh0`` are ``[N, H, *, *]``), and tokens past
    ``cu_seqlens[-1]`` get zero gradients, as in the Triton backward.
    ``prepare`` names the forward-recompute backend (see
    ``register_fwd_prepare``); None reads ``AITER_KDA_BWD_PREPARE``, default
    ``"triton"``. ``dhu_bv`` / ``fwd_h_bv`` override the V-block widths of
    ``dhu`` and the K5 recurrence. Unset tunables come from the tuned configs
    (``AITER_CONFIG_KDA_CHUNK_BWD``, see ``tuned_config``), then defaults.
    """

    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    stream = torch.cuda.current_stream()
    dummy, int32_dummy = _placeholder_pair(dev)
    f32 = {"device": dev, "dtype": torch.float32}
    varlen = cu_seqlens is not None
    if varlen:
        if B != 1:
            raise ValueError(f"varlen inputs must be flattened to B = 1, got B = {B}.")
        sched = _Varlen(cu_seqlens, chunk_indices)
        chunk_indices, NT, N = sched.chunk_indices, sched.NT, sched.N
        cu, ci, co = sched.cu, sched.ci, sched.co
    else:
        NT, N = (T + BT - 1) // BT, B
        cu = ci = co = int32_dummy
    NBH = B * H
    cfg = tuned_config(N, T // N if varlen else T, H, varlen)
    dhu_bv = dhu_bv or cfg.get("dhu_bv") or DEFAULT_DHU_BV
    fwd_h_bv = fwd_h_bv or cfg.get("fwd_h_bv")
    chunk_delta_attn_fwd_prepare = _fwd_prepare_fn(prepare, cfg.get("prepare"))
    # Varlen tokens past cu_seqlens[-1] are never written: zero those outputs.
    alloc = torch.zeros if varlen else torch.empty

    timer = _Timer()
    qn, kn, bn, gc, w, u, qg, kg, Aqk, Akk = chunk_delta_attn_fwd_prepare(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
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
    timer.mark("fwd_prepare")
    _check("fwd_prepare (Triton)", q=qn, k=kn, g=gc, w=w, u=u, qg=qg, kg=kg, Akk=Akk)
    h, v_new = kda_bwd_fwd_h_flydsl(
        kg, w, u, gc, initial_state, state_v_first, cu_seqlens, bv=fwd_h_bv
    )
    _check("fwd_h", h=h, v_new=v_new)
    timer.mark("fwd_h")
    do = do.contiguous()

    dAqk = alloc(B, T, H, BT, **f32)
    dv = alloc(v.shape, device=dev, dtype=v.dtype)
    _run_compiled(
        (compile_dav_mfma if _MFMA else compile_dav)(H, v.dtype, Aqk.dtype, varlen),
        v_new,
        Aqk,
        do,
        dAqk,
        dv,
        cu,
        ci,
        T,
        float(scale),
        NT,
        NBH,
        stream,
    )
    _check("dav", dAqk=dAqk, dv=dv)
    timer.mark("dav")

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = (
        torch.empty_like(initial_state, dtype=torch.float32)
        if initial_state is not None
        else None
    )
    dv2 = alloc(dv.shape, device=dev, dtype=dv.dtype)
    _run_compiled(
        (
            compile_dhu_mfma(
                H,
                kg.dtype,
                v.dtype,
                initial_state is not None,
                dht is not None,
                state_v_first,
                varlen,
                dhu_bv,
            )
            if _MFMA
            else compile_dhu(
                H,
                kg.dtype,
                v.dtype,
                initial_state is not None,
                dht is not None,
                state_v_first,
                varlen,
            )
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
        cu,
        co,
        T,
        float(scale),
        N * H,
        stream,
    )
    _check("dhu", dh=dh, dh0=dh0, dv2=dv2)
    timer.mark("dhu")

    dq = alloc(B, T, H, K, **f32)
    dk = alloc(B, T, H, K, **f32)
    dg = alloc(B, T, H, K, **f32)
    dv = alloc(v.shape, device=dev, dtype=v.dtype)
    db = alloc(B, T, H, **f32)
    dAkk = alloc(B, T, H, BT, **f32)
    _run_compiled(
        (compile_wy_dqkg_mfma if _MFMA else compile_wy_dqkg)(
            H, qn.dtype, v.dtype, bn.dtype, Akk.dtype, h.dtype, varlen
        ),
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
        cu,
        ci,
        T,
        float(scale),
        NT,
        NBH,
        stream,
    )
    _check("wy_dqkg", dq=dq, dk=dk, dv=dv, dg=dg, db=db, dAkk=dAkk)
    timer.mark("wy_dqkg")

    dq2 = alloc(q.shape, device=dev, dtype=q.dtype)
    dk2 = alloc(k.shape, device=dev, dtype=k.dtype)
    dg2 = alloc(B, T, H, K, **f32)
    db2 = alloc(beta.shape, device=dev, dtype=beta.dtype)
    _run_compiled(
        (compile_intra_mfma if _MFMA else compile_intra)(
            H,
            qn.dtype,
            bn.dtype,
            q.dtype,
            beta.dtype,
            q.dtype,
            beta.dtype,
            use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel,
            varlen,
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
        cu,
        ci,
        T,
        NT,
        NBH,
        stream,
    )
    _check("intra", dq=dq2, dk=dk2, dg=dg2, db=db2)
    timer.mark("intra")

    use_bias = use_gate_in_kernel and dt_bias is not None
    dA_part = torch.empty(NT, B, H, K, **f32) if use_gate_in_kernel else None
    db_part = torch.empty(NT, B, H, K, **f32) if use_bias else None
    dg_out = alloc(g.shape, device=dev, dtype=g.dtype)
    _run_compiled(
        compile_gate_cumsum(
            H,
            g.dtype,
            g.dtype,
            use_gate_in_kernel,
            use_bias,
            lower_bound is not None,
            varlen,
        ),
        dg2,
        g,
        A_log.float().contiguous() if use_gate_in_kernel else dummy,
        dt_bias.float().contiguous() if use_bias else dummy,
        dg_out,
        dA_part if dA_part is not None else dummy,
        db_part if db_part is not None else dummy,
        cu,
        ci,
        T,
        0.0 if lower_bound is None else float(lower_bound),
        NT,
        NBH,
        stream,
    )
    _check("gate_cumsum", dg=dg_out, dA_part=dA_part, db_part=db_part)
    timer.mark("gate_cumsum")
    timer.report(f"B={B} T={T} H={H} varlen={varlen}")
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
    cu_seqlens=None,
    lower_bound=None,
    safe_gate=None,
    prepare=None,
):
    """#8739 ``register_chunk_bwd`` contract: ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias)``.

    q, k, g ``[B, T, H, K]``, v and do ``[B, T, H, V]``, beta ``[B, T, H]``,
    no initial or final state; ``cu_seqlens`` (optional, ``B = 1``) packs
    variable-length sequences. With ``use_gate_in_kernel``, ``lower_bound``
    selects K3's safe gate ``g = lower_bound * sigmoid(exp(A_log) * (g +
    dt_bias))`` (``[-5, 0)``) instead of ``-exp(A_log) * softplus(g +
    dt_bias)``; ``safe_gate`` defaults to ``lower_bound is not None``.
    ``prepare`` selects a registered forward-recompute backend
    (``register_fwd_prepare``). q/k/v/do are computed in bf16; each gradient
    comes back in its input's dtype. Falls back to the Triton backward when
    the call is outside ``kda_bwd_flydsl_supported``.
    """
    in_dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype)
    bf = torch.bfloat16
    q, k, v, do = (t.to(bf).contiguous() for t in (q, k, v, do))
    g, beta = g.contiguous(), beta.contiguous()
    B, _, H, K = q.shape
    kw = {
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "use_gate_in_kernel": use_gate_in_kernel,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
        "lower_bound": lower_bound,
        "safe_gate": lower_bound is not None if safe_gate is None else safe_gate,
    }
    if kda_bwd_flydsl_supported(
        H=H,
        HV=v.shape[2],
        K=K,
        V=v.shape[-1],
        chunk_size=chunk_size,
        is_varlen=cu_seqlens is not None,
        dtype=q.dtype,
        B=B,
    ):
        out = chunk_kda_bwd_flydsl_full(
            do,
            None,
            q,
            k,
            v,
            g,
            beta,
            scale,
            chunk_size=chunk_size,
            prepare=prepare,
            **kw,
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

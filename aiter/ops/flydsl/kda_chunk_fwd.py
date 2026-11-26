# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL KDA (``chunk_kimi_delta_attn``) chunk forward.

Stages:

1. ``gate_cumsum`` -- chunk-local log2 cumsum, optionally through the in-kernel
   gate (``A_log`` / ``dt_bias``, softplus or the ``lower_bound`` sigmoid);
2. ``intra`` -- in-kernel q/k l2norm and beta sigmoid, ``Aqk``, the
   ``(I + Akk)^-1`` solve and ``w`` / ``u`` / ``qg`` / ``kg``;
3. ``h`` / ``v_new`` recurrence -- the GDN K5 kernel (``chunk_gated_delta_h``,
   per-channel ``USE_GK``, token-major ``w`` / ``u``);
4. ``output`` -- ``o = scale * qg h + Aqk v_new``.

Inputs and outputs are token-major ``[B, T, H, *]`` (fla ``chunk_kda``), the
layout of the fused KDA layer (#8739) and of the FlyDSL chunk backward
(#8740). ``chunk_kda_fwd_prepare_flydsl`` returns the same tuple as the
Triton ``chunk_delta_attn_fwd_prepare`` so the backward's recompute can use it.
"""

from __future__ import annotations

import torch

from .kernels.kda_chunk_fwd import (
    BT,
    compile_gate_cumsum,
    compile_intra,
    compile_output,
)
from .kernels.tensor_shim import _run_compiled
from .linear_attention_prefill_kernels import (
    _as_int32,
    _get_or_compile_opt,
    _hipeq_select_bv,
    _is_gfx942,
    _placeholder_pair,
    _resolve_prologue,
)


def kda_fwd_flydsl_supported(
    *, H: int, HV: int, K: int, V: int, chunk_size, dtype, is_varlen: bool = False
) -> bool:
    """Calls the FlyDSL forward serves; everything else stays on Triton.

    ``is_varlen`` is accepted for signature compatibility; varlen is served.
    """
    from aiter.jit.utils.chip_info import get_gfx_runtime

    # ``intra`` stages ~128 KB of LDS (gfx950 has 160 KB); K5 bakes BT=64, K=128.
    return (
        get_gfx_runtime() == "gfx950"
        and K == 128
        and V == 128
        and chunk_size in (None, BT)
        and H == HV
        and dtype == torch.bfloat16
    )


class _Chunks:
    """Chunk schedule of one call: dense ``B x NT`` or the varlen packing."""

    def __init__(self, B, T, device, cu_seqlens=None):
        _, self.int32_dummy = _placeholder_pair(device)
        self.varlen = cu_seqlens is not None
        if not self.varlen:
            self.N = B
            self.NT = (T + BT - 1) // BT  # chunks per sequence
            self.grid_t = self.NT
            self.max_seq_chunks = self.NT
            self.cbos = self.clen = self.int32_dummy
            self.kernel_cu = self.offsets = self.int32_dummy
            return
        if B != 1:
            raise ValueError(f"varlen needs B == 1, got B={B}.")
        # Same numbering as K5's ``chunk_offsets``: sequence n owns chunks
        # offsets[n] .. offsets[n] + ceil(len_n / BT) - 1.
        nt_total, offsets, kernel_cu, N, _ = _resolve_prologue(cu_seqlens, BT, 0, 0, T)
        bounds = cu_seqlens.tolist()
        bos, lens = [], []
        for n in range(N):
            for t0 in range(bounds[n], bounds[n + 1], BT):
                bos.append(t0)
                lens.append(min(BT, bounds[n + 1] - t0))
        if len(bos) != nt_total:
            raise ValueError(
                f"varlen chunk count {len(bos)} != K5 chunk count {nt_total}."
            )
        self.N = N
        self.NT = nt_total
        self.max_seq_chunks = max(
            ((bounds[n + 1] - bounds[n] + BT - 1) // BT for n in range(N)), default=0
        )
        self.grid_t = nt_total
        self.cbos = torch.tensor(bos, dtype=torch.int32, device=device)
        self.clen = torch.tensor(lens, dtype=torch.int32, device=device)
        self.kernel_cu = _as_int32(kernel_cu)
        self.offsets = _as_int32(offsets)


def chunk_kda_fwd_prepare_flydsl(
    q,
    k,
    v,
    g,
    beta,
    scale,
    *,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    store_qkb=True,
    cu_seqlens=None,
    chunks=None,
):
    """``(q, k, beta, g_cumsum, w, u, qg, kg, Aqk, Akk)``, as the Triton prepare.

    q/k/beta come back after their activations (``None`` when
    ``store_qkb=False``). g_cumsum, Aqk and Akk (the inverse) are fp32;
    w/u/qg/kg are bf16. All token-major; with ``cu_seqlens`` (B = 1) the
    chunks restart at every sequence start.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    if chunks is None:
        chunks = _Chunks(B, T, dev, cu_seqlens)
    varlen = chunks.varlen
    stream = torch.cuda.current_stream()
    dummy, _ = _placeholder_pair(dev)
    f32 = {"device": dev, "dtype": torch.float32}
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))

    gc = torch.empty(B, T, H, K, **f32)
    use_bias = use_gate_in_kernel and dt_bias is not None
    _run_compiled(
        compile_gate_cumsum(
            H, g.dtype, use_gate_in_kernel, use_bias, lower_bound is not None, varlen
        ),
        g,
        A_log.float().contiguous() if use_gate_in_kernel else dummy,
        dt_bias.float().contiguous() if use_bias else dummy,
        gc,
        chunks.cbos,
        chunks.clen,
        T,
        0.0 if lower_bound is None else float(lower_bound),
        chunks.grid_t,
        B * H,
        stream,
    )

    qn = torch.empty_like(q) if store_qkb else None
    kn = torch.empty_like(k) if store_qkb else None
    bn = torch.empty(B, T, H, **f32) if store_qkb else None
    bf = torch.bfloat16
    qg = torch.empty(B, T, H, K, device=dev, dtype=bf)
    kg = torch.empty_like(qg)
    w = torch.empty_like(qg)
    u = torch.empty(B, T, H, V, device=dev, dtype=bf)
    Aqk = torch.empty(B, T, H, BT, **f32)
    Akk = torch.empty_like(Aqk)
    _run_compiled(
        compile_intra(
            H,
            q.dtype,
            v.dtype,
            beta.dtype,
            use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel,
            store_qkb,
            varlen,
            # The lower_bound gate bounds the per-token decay, which the MFMA
            # diagonal tiles need; any other gate keeps them in fp32.
            use_gate_in_kernel and lower_bound is not None,
        ),
        q,
        k,
        v,
        gc,
        beta,
        qn if store_qkb else dummy,
        kn if store_qkb else dummy,
        bn if store_qkb else dummy,
        qg,
        kg,
        w,
        u,
        Aqk,
        Akk,
        chunks.cbos,
        chunks.clen,
        T,
        float(scale),
        chunks.grid_t,
        B * H,
        stream,
    )
    return qn, kn, bn, gc, w, u, qg, kg, Aqk, Akk


def kda_fwd_h_flydsl(
    kg,
    w,
    u,
    g_cumsum,
    initial_state,
    output_final_state,
    state_v_first,
    cu_seqlens=None,
    chunks=None,
):
    """K5 recurrence: ``h``, ``v_new [B, H, T, V]`` and the final state.

    ``h`` is ``[B, NT, H, V, K]`` (varlen: ``[1, NT_total, H, V, K]``). The
    initial / final states are per sequence, ``[N, H, K, V]`` (``[N, H, V, K]``
    when ``state_v_first``); the final state is fp32.
    """
    B, T, H, K = kg.shape
    V = u.shape[-1]
    if chunks is None:
        chunks = _Chunks(B, T, kg.device, cu_seqlens)
    N = chunks.N
    h0 = None
    if initial_state is not None:
        if initial_state.shape[0] != N:
            raise ValueError(
                f"initial_state has {initial_state.shape[0]} states for {N} sequences."
            )
        h0 = initial_state.to(torch.float32)
        if not state_v_first:
            h0 = h0.transpose(-1, -2)
        h0 = h0.contiguous()
    total_chunks = chunks.NT if chunks.varlen else B * chunks.NT
    BV = _hipeq_select_bv(kg.device, H, total_chunks, chunks.max_seq_chunks)
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
        bool(output_final_state),
        True,  # save_vn
        chunks.varlen,
        False,  # wu_contig: w/u are token-major
        state_bf16=False,
        g_log2_scaled=True,
        sched_gfx942=_is_gfx942(),
        bf16_convert_trunc=False,
        snapshot_bf16=True,
    )
    dummy, int32_dummy = _placeholder_pair(kg.device)
    h = kg.new_empty(B, chunks.NT, H, V, K)
    v_new = kg.new_empty(B, H, T, V)
    fs = (
        torch.empty(N, H, V, K, device=kg.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    _run_compiled(
        launch_fn,
        kg,
        u,
        w,
        v_new,
        dummy,
        g_cumsum,
        h,
        h0 if h0 is not None else dummy,
        fs if fs is not None else dummy,
        chunks.kernel_cu,
        chunks.offsets,
        int32_dummy,
        T,
        T,
        N,
        V // BV,
        N * H,
        torch.cuda.current_stream(),
    )
    if fs is not None and not state_v_first:
        fs = fs.transpose(-1, -2).contiguous()
    return h, v_new, fs


def chunk_kda_fwd_flydsl(
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
    lower_bound=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
):
    """FlyDSL KDA chunk forward: ``(o, final_state)``.

    q, k, g ``[B, T, H, 128]``; v ``[B, T, H, 128]``; beta ``[B, T, H]``; all
    token-major, same flags as ``chunk_kimi_delta_attn``. With ``cu_seqlens``
    (``[N + 1]``, B = 1) the N packed sequences run independently, and
    ``initial_state`` / the final state hold one state per sequence. q/k/v
    must be bf16; g and beta may be bf16 or fp32. ``o`` is bf16, the final
    state fp32. Raises ``ValueError`` outside ``kda_fwd_flydsl_supported``.
    """
    B, T, H, K = q.shape
    if not kda_fwd_flydsl_supported(
        H=H,
        HV=v.shape[2],
        K=K,
        V=v.shape[-1],
        chunk_size=chunk_size,
        dtype=q.dtype,
    ) or not (k.dtype == v.dtype == q.dtype):
        raise ValueError(
            "FlyDSL KDA forward needs gfx950, bf16 q/k/v, K=V=128, chunk 64 and "
            f"H==HV; got q={tuple(q.shape)} {q.dtype}, "
            f"v={tuple(v.shape)} {v.dtype}, chunk_size={chunk_size}."
        )
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    if not use_gate_in_kernel:
        lower_bound = None
    chunks = _Chunks(B, T, q.device, cu_seqlens)

    *_, gc, w, u, qg, kg, Aqk, _ = chunk_kda_fwd_prepare_flydsl(
        q,
        k,
        v,
        g,
        beta,
        scale,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        store_qkb=False,
        chunks=chunks,
    )
    h, v_new, fs = kda_fwd_h_flydsl(
        kg,
        w,
        u,
        gc,
        initial_state,
        output_final_state,
        state_v_first,
        chunks=chunks,
    )
    # Not empty_like(v): a permuted v view would hand the kernel its strides.
    o = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    _run_compiled(
        compile_output(H, h.dtype, v_new.dtype, o.dtype, chunks.varlen),
        qg,
        h,
        v_new,
        Aqk,
        o,
        chunks.cbos,
        chunks.clen,
        T,
        float(scale),
        chunks.grid_t,
        B * H,
        torch.cuda.current_stream(),
    )
    return o, fs

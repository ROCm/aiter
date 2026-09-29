# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared fixtures for the gated-delta-rule tests and benchmarks.

Input generation, the pure-PyTorch K5 reference, and the K5+K6 reference output
live here so that the correctness tests
(``test_flydsl_gdn_fused_k5k6.py``) and the perf sweep
(``op_benchmarks/flydsl/bench_gdn_fused_k5k6.py``) can share them.
"""

from __future__ import annotations

import torch
import triton

# Tolerance shared by every GDN correctness assert in this repo.
_RMSE_TOL = 5e-2


def ref_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
    g_head_major=False,
):
    """Reference in FP32 for correctness checking.

    This models K5 in isolation -- the inter-chunk state scan -- so it must
    match what the K5 kernel computes, NOT the full KDA/GDN algorithm. The two
    gate variants differ in whether K5 gates the chunk contribution:

      * Scalar gate (``g`` is not None, USE_G path, GDN/Qwen3-Next):
          b_v[t] = u[t] - w[t] @ h^T
          gate[t] = exp(g_last - g_cumsum[t])   # scalar, broadcast over V
          h = h * exp(g_last) + (b_v * gate)^T @ k[t]
        Here the per-token decay is a single scalar per row of ``b_v`` (broadcast
        over V), so K5 applies it directly: gate ``b_v``, decay ``h`` by
        ``exp(g_last)``.

      * Per-channel gate (``gk`` is not None, USE_GK path, KDA/Kimi-K3):
          b_v[t] = u[t] - w[t] @ h^T
          h[:, j] *= exp(gk_last[j])             # per-K-dim inter-chunk decay
          h = h + b_v^T @ k[t]                    # NB: b_v and k both un-gated
        There is intentionally no ``b_v`` (v_new) gate here. The KDA decay is a
        vector over the K dimension, not a scalar over V, so it multiplies ``k``
        rather than ``b_v``. In the full algorithm that intra-chunk k-gating is
        folded into ``k``/``w`` upstream by the prepare kernels (K1-K4); K5 owns
        ONLY the inter-chunk carry -- decay the running state per K-column, then
        accumulate the un-gated outer product. This reference and the K5 kernel
        both take ``k``/``w`` exactly as passed (the test does not pre-gate
        them), so the two agree by construction; the absent ``b_v`` gate is
        correct, not an omission.

    ``w`` and ``u`` are expected in token-major layout ``[B, T, H, *]``.
    """
    # At most one gate: ``g`` scalar (USE_G) or ``gk`` per-channel (USE_GK).
    # Both None is valid and means no gate decay (USE_G=False), matching the
    # kernel's pure padding-mask behaviour.
    assert g is None or gk is None, "g and gk are mutually exclusive"
    B, T, Hg_dim, K_dim = k.shape
    # ``gk`` is accepted flat ``[T, H, K]`` (varlen packs the batch away) or
    # token-major ``[B, T, H, K]``. Normalise to 4-D so the loop below can index
    # the batch the same way the ``g`` path does -- indexing it without a batch
    # term is exactly the bug this reference is used to catch in the kernel.
    if gk is not None and gk.dim() == 3:
        gk = gk.unsqueeze(0)
    if gk is not None:
        assert gk.shape[0] == B, (
            f"gk batch dim {gk.shape[0]} does not match k's B={B}; "
            f"gk.shape={tuple(gk.shape)}"
        )
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT_dim = chunk_size
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT_dim)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(triton.cdiv(int(seq_len), BT_dim) for seq_len in seq_lens)
    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, V_dim, K_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = (
        torch.zeros(N, H_dim, V_dim, K_dim, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [
                (s, cu_seqlens[s].item(), cu_seqlens[s + 1].item()) for s in range(N)
            ]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = triton.cdiv(seq_len, BT_dim)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                h_state = torch.zeros(
                    V_dim, K_dim, dtype=torch.float32, device=k.device
                )
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT_dim
                    t_end = min(t_start + BT_dim, seq_len)
                    actual_bt = t_end - t_start

                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start : bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start : bos + t_end, i_h].float()
                    b_v = u_chunk - w_chunk @ h_state.T
                    v_new_out[b_idx, bos + t_start : bos + t_end, i_h] = b_v

                    k_chunk = k[b_idx, bos + t_start : bos + t_end, i_hg].float()

                    if gk is not None:
                        # USE_GK (KDA): decay the carried state per K-column by
                        # exp(gk_last), then accumulate the outer product with
                        # both b_v and k un-gated. No per-token b_v gate exists
                        # on this path: the intra-chunk k-gating is the
                        # pipeline's job (K1-K4), not part of K5.
                        # gk_last is the gate at the chunk's last valid token,
                        # i.e. the cumulative decay across the whole chunk.
                        last_idx = bos + t_end - 1
                        gk_last = gk[b_idx, last_idx, i_h].float()  # [K]
                        h_state = h_state * torch.exp(gk_last).unsqueeze(0)
                        h_state = h_state + b_v.T @ k_chunk
                    else:
                        # USE_G: scalar h decay + v_new gating.
                        # g sequence for (batch b_idx, head i_h): g is always
                        # 3-D (or None). head-major [B,H,T] -> g[b_idx, i_h];
                        # token-major [B,T,H] -> g[b_idx, :, i_h].
                        if g is None:
                            g_seq = None
                        elif g_head_major:
                            g_seq = g[b_idx, i_h]
                        else:
                            g_seq = g[b_idx, :, i_h]

                        mask = torch.zeros(BT_dim, device=k.device)
                        mask[:actual_bt] = 1.0
                        if g_seq is None:
                            # No g: no gate decay; valid rows have gate=1 and
                            # padding rows are not in the chunk slice at all.
                            # Matches the kernel's pure padding masking under
                            # USE_G=False.
                            gate = mask[:actual_bt]
                        else:
                            last_idx = bos + t_end - 1
                            g_last = g_seq[last_idx].float()
                            g_chunk = g_seq[bos + t_start : bos + t_end].float()
                            gate = torch.where(
                                mask[:actual_bt].bool(),
                                torch.exp(g_last - g_chunk),
                                torch.zeros_like(g_chunk),
                            )
                            h_state = h_state * torch.exp(g_last)
                        b_v_gated = b_v * gate.unsqueeze(-1)
                        b_v_gated_cast = b_v_gated.to(k.dtype).float()
                        h_state = h_state + b_v_gated_cast.T @ k_chunk

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2).mean().sqrt()
    denom = b.float().pow(2).mean().sqrt() + 1e-8
    return (diff / denom).item()


def _make_cu_seqlens(seq_lens: list[int], device="cuda") -> torch.Tensor:
    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def _make_inputs(
    H, Hg, K, V, T_flat, seq_lens, gate, *, B=1, gk_decay=-0.1, device="cuda"
):
    """Build fused-forward inputs.

    Returns a dict with both token-major (reference) and head-major (kernel)
    layouts of w/u, plus q/k, the chosen gate tensor, h0, and cu_seqlens.

    ``B > 1`` is the dense multi-batch layout and is mutually exclusive with
    varlen: varlen packs every sequence into one batch via ``cu_seqlens``, so
    the batch axis only carries independent sequences when ``cu`` is None.
    Dense B>1 is what exercises the batch term in the kernel's gate addressing.
    """
    dtype = torch.bfloat16
    N = len(seq_lens)
    if B > 1 and N > 1:
        raise ValueError("dense B>1 and varlen (len(seq_lens)>1) are exclusive.")
    cu = _make_cu_seqlens(seq_lens, device) if N > 1 else None
    if cu is None:
        N = B  # dense: one independent sequence per batch

    q = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    w_tm = torch.randn(B, T_flat, H, K, dtype=dtype, device=device) * 0.1
    u_tm = torch.randn(B, T_flat, H, V, dtype=dtype, device=device) * 0.1
    w_hm = w_tm.permute(0, 2, 1, 3).contiguous()
    u_hm = u_tm.permute(0, 2, 1, 3).contiguous()

    g = gk = None
    if gate == "g":
        g = (
            (torch.randn(B, H, T_flat, dtype=torch.float32, device=device).abs() * -0.5)
            .cumsum(dim=2)
            .contiguous()
        )
    else:  # "gk"
        # ``gk_decay`` sets how fast the per-channel gate decays. The default
        # accumulates to exp(gk_last) ~ 0 over a few hundred tokens, which
        # saturates the decay: the output then barely moves when the gate is
        # perturbed, so a test using it cannot detect a *subtly* wrong gate
        # (only a missing one). Pass a smaller magnitude to keep exp(gk_last)
        # inside the sensitive range.
        gk = (
            (
                torch.randn(B, T_flat, H, K, dtype=torch.float32, device=device).abs()
                * gk_decay
            )
            .cumsum(dim=1)
            .contiguous()
        )

    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01

    # ``g`` is head-major [B, H, T_flat]; the kernel only checks the trailing
    # two dims. The pure-PyTorch reference indexes ``g[b_idx, i_h]`` and needs
    # the batch dim explicitly, so it takes the same tensor. At B == 1 the
    # kernel is equally happy with the 2-D [H, T_flat] spelling, so squeeze it
    # back down to keep exercising that path where it used to be covered.
    g_ref = g
    if g is not None and B == 1:
        g = g.squeeze(0).contiguous()

    # The HIP K5 fn takes ``gk`` flat as [T_flat, H, K] (no batch dim); the
    # FlyDSL and Triton paths accept the batched [B, T_flat, H, K] too. Expose
    # the flat spelling for callers that need the strict one -- only meaningful
    # for B == 1, which is where that flat contract is well-defined.
    gk_flat = None
    if gk is not None and B == 1:
        gk_flat = gk.squeeze(0).contiguous()

    return {
        "q": q,
        "k": k,
        "w_tm": w_tm,
        "u_tm": u_tm,
        "w_hm": w_hm,
        "u_hm": u_hm,
        "g": g,
        "g_ref": g_ref,
        "gk": gk,
        "gk_flat": gk_flat,
        "h0": h0,
        "cu": cu,
        "N": N,
        "B": B,
        "H": H,
        "Hg": Hg,
        "K": K,
        "V": V,
        "T_flat": T_flat,
    }


def _reference_o(inp, *, scale, use_exp2):
    """reference o via pure-PyTorch K5 ref then Triton K6.

    K5 ref produces token-major v_new [B, T, H, V] and h [B, NT, H, V, K];
    Triton K6 expects head-major v [B, H, T, V], so permute v_new. The K6 gate
    is scalar-``g`` only (KDA folds gk into K5), so pass g through and gk=None.
    """
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_fwd_o_opt_vk

    h_ref, v_new_ref, _ = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"],
        w=inp["w_tm"],
        u=inp["u_tm"],
        g=inp["g_ref"],
        gk=inp["gk"],
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        g_head_major=True,
    )
    # token-major [B, T, H, V] -> head-major [B, H, T, V]
    v_hm = v_new_ref.permute(0, 2, 1, 3).contiguous().to(inp["u_tm"].dtype)
    o = inp["u_tm"].new_empty(inp["u_tm"].shape)  # [B, T, H, V]
    chunk_fwd_o_opt_vk(
        q=inp["q"],
        k=inp["k"],
        v=v_hm,
        o=o,
        h=h_ref.to(inp["u_tm"].dtype),
        g=inp["g"],
        scale=scale,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
    )
    return o


def _pipeline_inputs(H, Hg, seq_lens, device="cuda"):
    K = V = 128
    T_flat = sum(seq_lens)
    N = len(seq_lens)
    dtype = torch.bfloat16
    cu = _make_cu_seqlens(seq_lens, device) if N > 1 else None
    q = torch.randn(1, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(1, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(1, T_flat, H, V, dtype=dtype, device=device) * 0.1
    g = torch.randn(1, T_flat, H, dtype=torch.float32, device=device) * -0.5
    beta = torch.randn(1, T_flat, H, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": K**-0.5,
        "initial_state": h0,
        "output_final_state": True,
        "cu_seqlens": cu,
    }

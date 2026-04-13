# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``causal_conv1d_update_fast`` / ``fused_reshape_causal_conv1d_update_fast``.

``causal_conv1d_update_fast`` updates ``conv_state`` in place; the reference mirrors
``_causal_conv1d_update_fast_kernel`` (non-APC), not ``causal_conv1d_update_ref``.
Shape extras that used to live in smoke tests are folded into
``test_causal_conv1d_update_fast_matches_ref`` (see ``_causal_conv1d_update_fast_ref_cases``).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.causal_conv1d import PAD_SLOT_ID
from aiter.ops.triton.causal_conv1d_update_fast import (
    causal_conv1d_update_fast,
    fused_reshape_causal_conv1d_update_fast,
)

cuda_ok = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device required"
)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ref_causal_conv1d_update_fast(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
    pad_slot_id: int | None,
) -> torch.Tensor:
    """Python port of ``_causal_conv1d_update_fast_kernel`` (non-APC, 1D indices).

    Mutates ``conv_state`` in place (like the Triton kernel). Clones ``x`` only for
    ``out`` leaves non-updated timesteps equal to the input.
    """
    out = x.clone()
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = width - 1
    np2 = triton.next_power_of_2(state_len)
    num_cache_lines = conv_state.shape[0]
    silu = activation in ("silu", "swish")

    if conv_state_indices.ndim != 1:
        raise NotImplementedError("reference supports 1D conv_state_indices only")

    for b in range(batch):
        coord_read = int(conv_state_indices[b].item())
        if pad_slot_id is not None and coord_read == pad_slot_id:
            continue
        coord_write = int(conv_state_indices[b].item())
        val = state_len - seqlen

        for f in range(dim):
            cols_hist = []
            for j in range(np2):
                if j < width - 1:
                    cols_hist.append(float(conv_state[coord_read, f, j].item()))
                else:
                    cols_hist.append(0.0)

            for j in range(np2):
                mask_cs = (coord_read < num_cache_lines) and (j + seqlen < state_len)
                v_cs = (
                    float(conv_state[coord_read, f, j + seqlen].item())
                    if mask_cs
                    else 0.0
                )
                t = j - val
                mask_x = (0 <= t) and (t < seqlen)
                v_x = float(x[b, f, t].item()) if mask_x else 0.0
                new_v = v_cs if mask_cs else v_x
                if j < state_len:
                    conv_state[coord_write, f, j] = torch.tensor(
                        new_v, dtype=conv_state.dtype, device=conv_state.device
                    )

            acc = 0.0
            for j in range(np2):
                wj = float(weight[f, j].item()) if j < width - 1 else 0.0
                acc += wj * cols_hist[j]
            w_last = float(weight[f, width - 1].item())
            x0 = float(x[b, f, 0].item())
            acc += w_last * x0
            if bias is not None:
                acc += float(bias[f].item())
            if silu:
                acc = acc / (1.0 + np.exp(-acc))
            out[b, f, 0] = torch.tensor(acc, dtype=out.dtype, device=out.device)

    return out


def _logical_feat_to_qkvz_col_v2(
    idx_feats: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    head_qkvz_dim: int,
    hv_ratio: int,
) -> int:
    nk, hk, hv = num_k_heads, head_k_dim, head_v_dim
    if idx_feats < nk * hk:
        h = idx_feats // hk
        r = idx_feats % hk
        return h * head_qkvz_dim + r
    if idx_feats < nk * hk * 2:
        rel = idx_feats - nk * hk
        h = rel // hk
        r = rel % hk
        return h * head_qkvz_dim + hk + r
    rel = idx_feats - nk * hk * 2
    gs = hv_ratio * hv
    h = rel // gs
    r = rel % gs
    return h * head_qkvz_dim + 2 * hk + r


def ref_fused_reshape_causal_conv1d_update_fast(
    x: torch.Tensor,
    num_actual_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    ba: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
    pad_slot_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: extract b/a/z like the kernel, build logical QKV, run ``ref_causal_conv1d_update_fast``."""
    num_tokens = x.shape[0]
    hv_ratio = num_v_heads // num_k_heads
    head_dim = head_k_dim + head_k_dim + head_v_dim * hv_ratio
    head_qkvz_dim = head_dim + head_v_dim * hv_ratio
    dim = num_k_heads * head_dim
    seqlen = x.shape[2]
    device = x.device
    dtype = x.dtype

    b_out = torch.empty(num_actual_tokens, num_v_heads, device=device, dtype=ba.dtype)
    a_out = torch.empty_like(b_out)
    for idx_seq in range(num_actual_tokens):
        for idx_hv in range(num_v_heads):
            idx_h = idx_hv // hv_ratio
            idx_v = idx_hv % hv_ratio
            b_off = idx_h * (2 * hv_ratio) + idx_v
            a_off = idx_h * (2 * hv_ratio) + hv_ratio + idx_v
            b_out[idx_seq, idx_hv] = ba[idx_seq, b_off]
            a_out[idx_seq, idx_hv] = ba[idx_seq, a_off]

    z_flat = z_out.reshape(num_tokens, -1).clone()
    core_flat = core_attn_out.reshape(num_tokens, -1).clone()
    gs = hv_ratio * head_v_dim
    for idx_seq in range(num_tokens):
        for idx_z in range(num_v_heads * head_v_dim):
            idx_z_x = (
                (idx_z // gs) * head_qkvz_dim
                + 2 * head_k_dim
                + hv_ratio * head_v_dim
                + (idx_z % gs)
            )
            z_flat[idx_seq, idx_z] = x[idx_seq, idx_z_x, 0]

    n_repeat = (num_tokens - 1) // num_actual_tokens if num_actual_tokens else 0
    for idx_repeat in range(n_repeat):
        for idx_seq in range(num_actual_tokens):
            idx_seq_remain = num_actual_tokens * (1 + idx_repeat) + idx_seq
            if idx_seq_remain < num_tokens:
                z_flat[idx_seq_remain].zero_()
                core_flat[idx_seq_remain].zero_()

    x_lin = torch.zeros(num_actual_tokens, dim, seqlen, device=device, dtype=dtype)
    for b in range(num_actual_tokens):
        for f in range(dim):
            col = _logical_feat_to_qkvz_col_v2(
                f, num_k_heads, head_k_dim, head_v_dim, head_qkvz_dim, hv_ratio
            )
            for t in range(seqlen):
                x_lin[b, f, t] = x[b, col, t]

    cs = conv_state.clone()
    if conv_state_indices is None:
        cidx = torch.arange(num_actual_tokens, device=device, dtype=torch.int32)
    else:
        cidx = conv_state_indices
    out_lin = ref_causal_conv1d_update_fast(
        x_lin,
        cs,
        weight,
        bias,
        activation,
        cidx,
        pad_slot_id,
    )
    if seqlen == 1:
        out_lin = out_lin.squeeze(-1)
    return (
        out_lin,
        b_out,
        a_out,
        z_flat.view_as(z_out),
        core_flat.view_as(core_attn_out),
        cs,
    )


def _causal_conv1d_update_fast_ref_cases():
    """Cartesian core grid plus former smoke shapes (width=3, small dim, seqlen=2)."""
    out = []
    for itype in (torch.float32, torch.bfloat16):
        for silu_activation in (True, False):
            for has_bias in (True, False):
                for seqlen in (1, 8):
                    for width in (2, 4):
                        out.append(
                            pytest.param(
                                1,
                                1024,
                                width,
                                seqlen,
                                itype,
                                silu_activation,
                                has_bias,
                                id=f"b1-d1024-w{width}-s{seqlen}-"
                                f"silu{silu_activation}-bias{has_bias}-"
                                f"{'fp32' if itype == torch.float32 else 'bf16'}",
                            )
                        )
    out.extend(
        [
            pytest.param(
                2,
                64,
                3,
                1,
                torch.bfloat16,
                True,
                True,
                id="smoke-b2-d64-w3-s1-bf16",
            ),
            pytest.param(
                1,
                128,
                4,
                2,
                torch.bfloat16,
                True,
                True,
                id="smoke-b1-d128-w4-s2-bf16",
            ),
        ]
    )
    return out


@cuda_ok
@pytest.mark.parametrize(
    (
        "batch",
        "dim",
        "width",
        "seqlen",
        "itype",
        "silu_activation",
        "has_bias",
    ),
    _causal_conv1d_update_fast_ref_cases(),
)
def test_causal_conv1d_update_fast_matches_ref(
    batch, dim, width, seqlen, itype, silu_activation, has_bias
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 6e-2
    seed_everything(0)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_tr = x.clone()
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=itype)
    conv_tr = conv_state.clone()
    conv_ref = conv_state.clone()
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    activation = None if not silu_activation else "silu"
    cidx = torch.arange(batch, dtype=torch.int32, device=device)

    out_ref = ref_causal_conv1d_update_fast(
        x, conv_ref, weight, bias, activation, cidx, PAD_SLOT_ID
    )
    out_tr = causal_conv1d_update_fast(
        x_tr,
        conv_tr,
        weight,
        bias,
        activation=activation,
        conv_state_indices=cidx,
        pad_slot_id=PAD_SLOT_ID,
    )
    torch.testing.assert_close(conv_tr, conv_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out_tr, out_ref, rtol=rtol, atol=atol)


@cuda_ok
@pytest.mark.parametrize(
    "num_k_heads,num_v_heads,head_k_dim,head_v_dim,num_tokens,num_actual_tokens,width",
    [
        (2, 2, 8, 8, 4, 2, 3),
        (2, 4, 16, 8, 6, 3, 4),
    ],
)
def test_fused_reshape_causal_conv1d_update_fast_matches_ref(
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    num_tokens,
    num_actual_tokens,
    width,
):
    device = "cuda"
    torch.manual_seed(1)
    hv_ratio = num_v_heads // num_k_heads
    assert hv_ratio * num_k_heads == num_v_heads
    head_dim = head_k_dim + head_k_dim + head_v_dim * hv_ratio
    head_qkvz_dim = head_dim + head_v_dim * hv_ratio
    qkvz_dim = num_k_heads * head_qkvz_dim
    dim = num_k_heads * head_dim
    seqlen = 1
    dtype = torch.bfloat16
    rtol, atol = 1e-2, 6e-2

    x = torch.randn(num_tokens, qkvz_dim, seqlen, device=device, dtype=dtype)
    ba = torch.randn(num_tokens, 2 * num_v_heads, device=device, dtype=dtype)
    z_out = torch.zeros(num_tokens, num_v_heads, head_v_dim, device=device, dtype=dtype)
    core = torch.zeros_like(z_out)
    conv_state = torch.randn(
        num_actual_tokens, dim, width - 1, device=device, dtype=dtype
    )
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    z_ref = z_out.clone()
    core_ref = core.clone()
    cs_ref_init = conv_state.clone()
    out_ref, b_ref, a_ref, z_r, c_r, cs_ref = (
        ref_fused_reshape_causal_conv1d_update_fast(
            x,
            num_actual_tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            ba,
            z_ref,
            core_ref,
            cs_ref_init,
            weight,
            bias,
            "silu",
            None,
            PAD_SLOT_ID,
        )
    )

    z_tr = z_out.clone()
    core_tr = core.clone()
    cs_tr = conv_state.clone()
    out_tr, b_tr, a_tr = fused_reshape_causal_conv1d_update_fast(
        x,
        num_actual_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ba,
        z_tr,
        core_tr,
        cs_tr,
        weight,
        bias,
        activation="silu",
        conv_state_indices=None,
        pad_slot_id=PAD_SLOT_ID,
    )

    torch.testing.assert_close(out_tr.float(), out_ref.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(b_tr, b_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a_tr, a_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(z_tr, z_r, rtol=rtol, atol=atol)
    torch.testing.assert_close(core_tr, c_r, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cs_tr, cs_ref, rtol=0.0, atol=0.0)

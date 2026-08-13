# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Accuracy comparison: opus_gdn HIP kernel vs Triton chunk_gated_delta_rule vs recurrent ref

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.opus_gdn_wu_prefill import opus_gdn_wu_prefill_fwd
from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    assert_close,
    device,
)
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = (
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (q, k, v, beta, g)
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.float()
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(*t, id=f"B{t[0]}-T{t[1]}-H{t[2]}-BT{t[3]}")
        for t in [
            (1, 64, 4, 64),
            (2, 128, 4, 64),
            (2, 256, 8, 64),
            (1, 512, 4, 64),
            (2, 128, 4, 16),
            (1, 256, 8, 16),
        ]
    ],
)
def test_opus_vs_triton(B: int, T: int, H: int, BT: int):
    torch.manual_seed(42)
    D = 128

    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    opus_o, _ = opus_gdn_wu_prefill_fwd(q, k, v, g, beta.to(torch.bfloat16), BT=BT)

    tri_o, _ = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.to(torch.bfloat16).clone(),
    )

    assert_close("opus_vs_tri_o", tri_o, opus_o, 0.01)


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(*t, id=f"B{t[0]}-T{t[1]}-H{t[2]}-BT{t[3]}")
        for t in [
            (1, 64, 4, 64),
            (2, 128, 4, 64),
            (2, 256, 8, 64),
            (2, 128, 4, 16),
        ]
    ],
)
def test_opus_vs_ref(B: int, T: int, H: int, BT: int):
    torch.manual_seed(42)
    D = 128

    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    opus_o, _ = opus_gdn_wu_prefill_fwd(q, k, v, g, beta.to(torch.bfloat16), BT=BT)

    ref_o, _ = recurrent_gated_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.to(torch.bfloat16).clone(),
        g=g.clone(),
    )

    assert_close("opus_vs_ref_o", ref_o, opus_o, 0.01)


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(*t, id=f"B{t[0]}-T{t[1]}-H{t[2]}-BT{t[3]}")
        for t in [
            (1, 128, 4, 64),
            (2, 256, 4, 64),
            (1, 128, 4, 16),
        ]
    ],
)
def test_opus_with_initial_state(B: int, T: int, H: int, BT: int):
    torch.manual_seed(42)
    D = 128

    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.1

    # opus expects [B,H,V,K], ref expects [B,H,K,V]
    h0_vk = h0.clone()
    h0_kv = h0.transpose(-1, -2).contiguous()

    opus_o, opus_ht = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta.to(torch.bfloat16),
        initial_state=h0_vk,
        output_final_state=True,
        BT=BT,
    )

    ref_o, ref_ht = recurrent_gated_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.to(torch.bfloat16).clone(),
        g=g.clone(),
        initial_state=h0_kv,
        output_final_state=True,
    )

    assert_close("init_o", ref_o, opus_o, 0.01)
    # opus_ht is [B,H,V,K], ref_ht is [B,H,K,V]
    assert_close("init_ht", ref_ht, opus_ht.transpose(-1, -2).contiguous(), 0.01)


@pytest.mark.parametrize("k2_mode", [1, 2], ids=["WF", "WS"])
@pytest.mark.parametrize("with_state", [False, True], ids=["no-state", "state"])
def test_opus_wu_forced_dense_paths(k2_mode: int, with_state: bool):
    """Exercise both final dense W/U families through their explicit ABI."""
    torch.manual_seed(20260731 + 10 * k2_mode + int(with_state))
    B, T, H, D = 1, 128, 4, 128

    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    h0_vk = (
        torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.1
        if with_state
        else None
    )

    out = torch.empty_like(v)
    opus_o, opus_ht = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0_vk,
        output_final_state=with_state,
        BT=64,
        BV=64,
        num_warps=8,
        k1_algo=1,
        k2_mode=k2_mode,
        out=out,
    )
    ref_o, ref_ht = recurrent_gated_delta_rule_ref(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        initial_state=(h0_vk.transpose(-1, -2).contiguous() if with_state else None),
        output_final_state=with_state,
    )

    assert opus_o.data_ptr() == out.data_ptr()
    assert_close("forced_wu_o", ref_o, opus_o, 0.01)
    if with_state:
        assert_close(
            "forced_wu_ht",
            ref_ht,
            opus_ht.transpose(-1, -2).contiguous(),
            0.01,
        )


@pytest.mark.parametrize("k2_mode", [1, 2], ids=["WF", "WS"])
def test_opus_wu_production_mode_ignores_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    k2_mode: int,
):
    """Production mode must ignore every benchmark-only launcher override."""
    for name in (
        "OPUS_GDN_SPLIT_THRESHOLD",
        "OPUS_GDN_OUT_VARIANT",
        "OPUS_GDN_SPLIT_NW",
        "OPUS_GDN_HIP_SCAN",
        "OPUS_GDN_SCAN32",
        "OPUS_GDN_REF",
        "OPUS_GDN_OUT_BV",
        "OPUS_GDN_OUT_NW",
        "OPUS_GDN_WF_VARIANT",
    ):
        monkeypatch.setenv(name, "999")

    torch.manual_seed(20260803 + k2_mode)
    B, T, H, D = 1, 64, 4, 128
    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    opus_o, _ = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        BT=64,
        BV=64,
        num_warps=8,
        k1_algo=1,
        k2_mode=k2_mode,
        use_env_overrides=False,
    )
    ref_o, _ = recurrent_gated_delta_rule_ref(q, k, v, beta, g)

    assert_close("production_wu_o", ref_o, opus_o, 0.01)


def test_scan32_override_rejects_gfx942(monkeypatch: pytest.MonkeyPatch):
    properties = torch.cuda.get_device_properties(device)
    gfx = properties.gcnArchName.split(":", 1)[0]
    if gfx != "gfx942":
        pytest.skip(f"scan32 architecture guard requires gfx942, got {gfx}")

    monkeypatch.setenv("OPUS_GDN_REF", "0")
    monkeypatch.setenv("OPUS_GDN_SCAN32", "1")

    torch.manual_seed(20260810)
    B, T, H, D = 1, 64, 4, 128
    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    with pytest.raises(RuntimeError, match="OPUS_GDN_SCAN32 requires gfx950"):
        opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            BT=64,
            BV=64,
            num_warps=8,
            k1_algo=1,
            k2_mode=2,
        )

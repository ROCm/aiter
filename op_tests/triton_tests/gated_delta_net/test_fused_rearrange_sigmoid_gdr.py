# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gated_delta_net import fused_rearrange_sigmoid_gated_delta_rule

cuda_ok = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device required"
)


def _softplus(x: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    return torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
        x,
    )


def ref_fused_rearrange_sigmoid_gdr(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float reference for decode path (B=1, one sequence), including GQA (HV >= H).

    KDA: pass ``a`` as [T, HV, K] and ``dt_bias`` as [HV, K]; the gate ``g`` is then
    a per-channel [K] vector that broadcasts over the [V, K] state rows.
    """
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    if HV % H != 0:
        raise ValueError(f"reference expects HV divisible by H, got H={H}, HV={HV}")
    group = HV // H
    B = 1
    o = torch.empty(B, T, HV, V, dtype=torch.float32, device=qkv.device)
    h_state = torch.zeros(HV, V, K, dtype=torch.float32, device=qkv.device)
    if initial_state is not None:
        h_state = initial_state[0].to(torch.float32).clone()

    for t in range(T):
        row = qkv[t]
        for hv in range(HV):
            i_h = hv // group
            q_vec = row[i_h * K : (i_h + 1) * K].float()
            k_vec = row[H * K + i_h * K : H * K + (i_h + 1) * K].float()
            v_vec = row[2 * H * K + hv * V : 2 * H * K + (hv + 1) * V].float()
            b_gate = b[t, hv].float()
            x = a[t, hv].float() + dt_bias[hv].float()
            sp = _softplus(x, beta, threshold)
            g = -torch.exp(A_log[hv].float()) * sp
            beta_out = torch.sigmoid(b_gate)
            if use_qk_l2norm_in_kernel:
                q_vec = q_vec * torch.rsqrt((q_vec * q_vec).sum() + 1e-6)
                k_vec = k_vec * torch.rsqrt((k_vec * k_vec).sum() + 1e-6)
            q_vec = q_vec * scale
            h_sub = h_state[hv]
            h_sub = h_sub * torch.exp(g)
            v_adj = v_vec - (h_sub * k_vec.unsqueeze(0)).sum(dim=-1)
            v_adj = v_adj * beta_out
            h_sub = h_sub + v_adj.unsqueeze(-1) * k_vec.unsqueeze(0)
            out_vec = (h_sub * q_vec.unsqueeze(0)).sum(dim=-1)
            o[0, t, hv] = out_vec
            h_state[hv] = h_sub
    return o, h_state.unsqueeze(0)


# Shapes aligned with ``test_gated_delta_rule.test_fused_recurrent``; dtypes are
# half-precision only — long packed ``T`` with float32 activations tends to blow
# up the recurrent reference / kernel without tighter dynamic-range clamps.
# Each row ends with ``use_qk_l2norm_in_kernel`` (True for stable long-T sweep).
# One small bf16 row uses False to cover the no–L2-norm path (replaces former ``basic``).
_FUSED_GDR_SWEEP = [
    (63, 1, 1, 64, 1, 1, torch.float16, True),
    (500, 4, 4, 60, 1, 1, torch.float16, True),
    (1000, 2, 8, 128, 1, 0.1, torch.float16, True),
    (1024, 2, 2, 128, 0.1, 1, torch.float16, True),
    (1024, 3, 3, 128, 1, 10, torch.float16, True),
    (2048, 4, 4, 64, 0.1, 1, torch.float16, True),
    (1024, 4, 4, 128, 1, 0.1, torch.float16, True),
    (1024, 4, 8, 128, 1, 10, torch.float16, True),
    (1024, 4, 4, 128, 1, 0.1, torch.bfloat16, True),
    (1024, 4, 8, 128, 1, 1, torch.bfloat16, True),
    (2048, 4, 8, 64, 0.1, 1, torch.bfloat16, True),
    (8, 4, 4, 16, 16**-0.5, 1, torch.bfloat16, False),
]


@cuda_ok
@pytest.mark.parametrize(
    (
        "T",
        "H",
        "HV",
        "D",
        "scale",
        "gate_logit_normalizer",
        "dtype",
        "use_qk_l2norm_in_kernel",
    ),
    [
        pytest.param(
            *row,
            id="T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}-l2{}".format(*row),
        )
        for row in _FUSED_GDR_SWEEP
    ],
)
def test_fused_rearrange_sigmoid_gdr_sweep(
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
):
    """Shape/dtype sweep aligned with ``test_gated_delta_rule.test_fused_recurrent``."""
    if HV % H != 0:
        pytest.skip("reference/kernel GQA mapping needs HV divisible by H")
    device = "cuda"
    K = V = D
    key_dim = H * K
    value_dim = HV * V

    if use_qk_l2norm_in_kernel:
        torch.manual_seed(42)
        qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype) * 0.05
        A_log = (
            torch.randn(HV, device=device, dtype=torch.float32).clamp(-2.0, 0.5) * 0.02
        )
        a = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(-1.0, 1.0)
        a = a / gate_logit_normalizer
        b_gate = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(
            -1.0, 1.0
        )
        dt_bias = (torch.randn(HV, device=device, dtype=dtype) * 0.005).clamp(-0.5, 0.5)
        initial = torch.randn(1, HV, V, K, device=device, dtype=dtype) * 0.05
    else:
        torch.manual_seed(0)
        qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype)
        A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.02
        a = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
        a = a / gate_logit_normalizer
        b_gate = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
        dt_bias = torch.randn(HV, device=device, dtype=dtype) * 0.01
        initial = torch.randn(1, HV, V, K, device=device, dtype=dtype)

    o_ref, h_ref = ref_fused_rearrange_sigmoid_gdr(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        1.0,
        20.0,
        scale,
        initial,
        use_qk_l2norm_in_kernel,
    )

    core = torch.empty(T, HV, V, device=device, dtype=dtype)
    o_tr, h_tr = fused_rearrange_sigmoid_gated_delta_rule(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        beta=1.0,
        threshold=20.0,
        scale=scale,
        initial_state=initial,
        inplace_final_state=False,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=False,
        core_attn_out=core,
    )

    if dtype == torch.bfloat16:
        rtol, atol = 0.05, 0.1
    elif dtype == torch.float16:
        rtol, atol = 0.03, 0.08
    else:
        rtol, atol = 0.02, 0.05

    if use_qk_l2norm_in_kernel:
        assert torch.isfinite(o_tr.float()).all(), "non-finite Triton output"
        assert torch.isfinite(h_tr.float()).all(), "non-finite Triton final_state"
    torch.testing.assert_close(o_tr.float(), o_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(h_tr[-1].float(), h_ref[0], rtol=rtol, atol=atol)


def _make_kda_inputs(T, H, HV, D, dtype, num_state_slots=1, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    K = V = D
    qkv = torch.randn(T, 2 * H * K + HV * V, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.5
    # KDA: one gate logit per (token, value head, key channel).
    a = torch.randn(T, HV, K, device=device, dtype=dtype)
    b_gate = torch.randn(T, HV, device=device, dtype=dtype)
    dt_bias = torch.randn(HV, K, device=device, dtype=dtype) * 0.5
    state = torch.randn(num_state_slots, HV, V, K, device=device, dtype=dtype)
    return qkv, A_log, a, b_gate, dt_bias, state


# KDA (per-channel gate). T >= 2 so every token after the first reads its own
# gate row; the last row uses production-sized heads (32 x 128).
_KDA_SHAPES = [
    (2, 1, 1, 16, torch.float32),
    (4, 2, 4, 64, torch.bfloat16),
    (8, 32, 32, 128, torch.bfloat16),
]


def _kda_tol(dtype):
    return (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)


@cuda_ok
@pytest.mark.parametrize(
    ("T", "H", "HV", "D", "dtype"),
    [pytest.param(*row, id="T{}-H{}-HV{}-D{}-{}".format(*row)) for row in _KDA_SHAPES],
)
def test_fused_rearrange_sigmoid_gdr_kda(
    T: int, H: int, HV: int, D: int, dtype: torch.dtype
):
    K = V = D
    qkv, A_log, a, b_gate, dt_bias, initial = _make_kda_inputs(T, H, HV, D, dtype)
    scale = K**-0.5

    o_ref, h_ref = ref_fused_rearrange_sigmoid_gdr(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        H * K,
        HV * V,
        K,
        V,
        1.0,
        20.0,
        scale,
        initial,
        True,
    )
    o_tr, h_tr = fused_rearrange_sigmoid_gated_delta_rule(
        A_log,
        a.view(T, HV * K),
        b_gate,
        dt_bias,
        qkv,
        H * K,
        HV * V,
        K,
        V,
        scale=scale,
        initial_state=initial,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
    )

    rtol, atol = _kda_tol(dtype)
    torch.testing.assert_close(o_tr.float(), o_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(h_tr[-1].float(), h_ref[0], rtol=rtol, atol=atol)


# Varlen KDA with per-token in-place state slots (speculative-decode verify layout).
_KDA_VARLEN = [
    ([2, 2], 1, 2, 16, torch.float32),
    ([1, 3, 2], 2, 4, 64, torch.bfloat16),
    ([4, 4, 4, 4], 32, 32, 128, torch.bfloat16),
]


@cuda_ok
@pytest.mark.parametrize(
    ("seq_lens", "H", "HV", "D", "dtype"),
    [
        pytest.param(
            *row,
            id="lens{}-H{}-HV{}-D{}-{}".format("_".join(map(str, row[0])), *row[1:]),
        )
        for row in _KDA_VARLEN
    ],
)
def test_fused_rearrange_sigmoid_gdr_kda_varlen(
    seq_lens: list[int], H: int, HV: int, D: int, dtype: torch.dtype
):
    device = "cuda"
    K = V = D
    T = sum(seq_lens)
    N = len(seq_lens)
    max_len = max(seq_lens)
    scale = K**-0.5
    # Slot 0 is unused; every (sequence, token) writes its own state slot.
    qkv, A_log, a, b_gate, dt_bias, pool = _make_kda_inputs(
        T, H, HV, D, dtype, num_state_slots=T + 1
    )
    pool_init = pool.clone()
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    ssm_state_indices = torch.full((N, max_len), -1, device=device, dtype=torch.int32)
    for n, (bos, L) in enumerate(zip(cu_seqlens[:-1].tolist(), seq_lens)):
        ssm_state_indices[n, :L] = torch.arange(bos + 1, bos + 1 + L)

    o_tr, _ = fused_rearrange_sigmoid_gated_delta_rule(
        A_log,
        a.view(T, HV * K),
        b_gate,
        dt_bias,
        qkv,
        H * K,
        HV * V,
        K,
        V,
        scale=scale,
        initial_state=pool,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
    )

    rtol, atol = _kda_tol(dtype)
    for n, (bos, L) in enumerate(zip(cu_seqlens[:-1].tolist(), seq_lens)):
        eos = bos + L
        o_ref, h_ref = ref_fused_rearrange_sigmoid_gdr(
            A_log,
            a[bos:eos],
            b_gate[bos:eos],
            dt_bias,
            qkv[bos:eos],
            H * K,
            HV * V,
            K,
            V,
            1.0,
            20.0,
            scale,
            pool_init[ssm_state_indices[n, 0]].unsqueeze(0),
            True,
        )
        torch.testing.assert_close(
            o_tr[0, bos:eos].float(), o_ref[0], rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            pool[ssm_state_indices[n, L - 1]].float(), h_ref[0], rtol=rtol, atol=atol
        )

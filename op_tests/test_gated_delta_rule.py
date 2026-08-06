# Copyright (C) 2023-2026, Songlin Yang, Yu Zhang

import os

os.environ.setdefault("AITER_TRITON_ONLY", "1")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from aiter.ops.chunk_gated_delta_rule_fwd_h import (
    chunk_gated_delta_rule_fwd_h_hip_fn,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    IS_AMD,
    IS_INTEL_ALCHEMIST,
    assert_close,
    device,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill import (
    chunk_gated_delta_rule_fwd_h_opt_vk,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
    build_gated_delta_rule_prefill_metadata,
)
from aiter.ops.triton.gated_delta_net import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_opt,
    chunk_gated_delta_rule_opt_vk,
    fused_recurrent_gated_delta_rule,
)


def _is_gfx12_runtime() -> bool:
    if not IS_AMD:
        return False
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(props, "gcnArchName", "")
        return arch.split(":")[0].startswith("gfx12") if arch else False
    except Exception:  # noqa: BLE001
        return False


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
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


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in [q, k, v, beta, g]
    )

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, beta, g = (x.to(torch.float32) for x in [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, seq_len, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert seq_len % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    q, k, v, k_beta, decay = (
        rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size)
        for x in [q, k, v, k_beta, decay.unsqueeze(-1)]
    )
    decay = decay.squeeze(-1).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn  # noqa: PLW0127
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for i in range(seq_len // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, "b h n c d -> b h (n c) d")
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 4, 60, 1, 1, torch.float),
            (2, 1000, 2, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 0.1, 1, torch.float),
            (4, 1024, 3, 3, 128, 1, 10, torch.float),
            (4, 2048, 4, 4, 64, 0.1, 1, torch.float),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 1, 10, torch.float16),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.bfloat16),
            (2, 1024, 4, 8, 128, 1, 1, torch.bfloat16),
            (4, 2048, 4, 8, 64, 0.1, 1, torch.bfloat16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    g = g / gate_logit_normalizer
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_() for x in (q, k, v, beta, g, h0)
    )
    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(
            repeat(q.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
        ).to(dtype),
        k=F.normalize(
            repeat(k.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
        ).to(dtype),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )
    # Use higher tolerance for bfloat16 due to lower precision
    tol = 0.005 if dtype == torch.bfloat16 else 0.002
    assert_close("o", ref, tri, tol)
    assert_close("ht", ref_ht, tri_ht, tol)


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(
                *test
            ),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.float16),
            # bfloat16 tests
            (2, 500, 3, 60, 1, 1, 0, False, torch.bfloat16),
            (2, 1000, 3, 64, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.bfloat16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.bfloat16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(
            reason="chunk_gated_delta_rule is not supported on alchemist for D>128"
        )

    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(True) for x in (q, k, v, beta, g, h0)
    )

    tri, tri_ht = chunk_gated_delta_rule(
        q=(
            F.normalize(q.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else q.clone()
        ),
        k=(
            F.normalize(k.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else k.clone()
        ),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    # do = torch.randn_like(v)
    # dht = torch.randn_like(h0)

    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    # ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    # ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    # assert_close('dq', ref_dq, tri_dq, 0.008)
    # assert_close('dk', ref_dk, tri_dk, 0.008)
    # assert_close('dv', ref_dv, tri_dv, 0.008)
    # assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    # assert_close('dg', ref_dg, tri_dg, 0.02)
    # assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 4, 60, 1, 1, torch.float),
            (2, 1000, 2, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 0.1, 1, torch.float),
            (4, 1024, 3, 3, 128, 1, 10, torch.float),
            (4, 2048, 4, 4, 64, 0.1, 1, torch.float),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 1, 10, torch.float16),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.bfloat16),
            (2, 1024, 4, 8, 128, 1, 1, torch.bfloat16),
            (4, 2048, 4, 8, 64, 0.1, 1, torch.bfloat16),
        ]
    ],
)
def test_fused_sigmoid_gating_delta_rule_update(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    """Test fused sigmoid gating delta rule update kernel."""
    torch.manual_seed(42)

    # Create input tensors
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    b = torch.randn(B, T, HV, dtype=dtype)  # beta logits

    # Gating parameters
    A_log = torch.randn(HV, dtype=torch.float32)
    a = torch.randn(B, T, HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Initial state
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    initial_state_indices = torch.arange(B, dtype=torch.long)

    # Move to device
    q, k, v, b, A_log, a, dt_bias, h0, initial_state_indices = (
        x.to(device) for x in (q, k, v, b, A_log, a, dt_bias, h0, initial_state_indices)
    )

    # Compute reference using recurrent implementation
    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    # This is already in log-space (negative values for decay)
    x = a + dt_bias[None, None, :]
    softplus_x = F.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
    g = -torch.exp(A_log[None, None, :]) * softplus_x
    beta = torch.sigmoid(b)

    # Expand q and k to match HV heads for reference implementation
    q_expanded = F.normalize(
        repeat(q.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
    ).to(dtype)
    k_expanded = F.normalize(
        repeat(k.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
    ).to(dtype)

    ref, _ = recurrent_gated_delta_rule_ref(
        q=q_expanded,
        k=k_expanded,
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Compute using fused kernel
    tri = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        b=b.clone(),
        initial_state_source=h0.clone(),
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    # Use higher tolerance for bfloat16 due to lower precision
    tol = 0.005 if dtype == torch.bfloat16 else 0.002
    assert_close("o", ref, tri, tol)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
            # bfloat16 tests
            (4, 60, 0, [0, 15], torch.bfloat16),
            (4, 64, 0, [0, 256, 500, 1000], torch.bfloat16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(
            reason="chunk_gated_delta_rule is not supported on alchemist for D>128"
        )
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(False) for x in (q, k, v, beta, g, h0)
    )
    # do = torch.randn_like(v)
    # dht = torch.rand_like(h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    # ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    # tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    # q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
            q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    # ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    # ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    # assert_close('dq', ref_dq, tri_dq, 0.007)
    # assert_close('dk', ref_dk, tri_dk, 0.008)
    # assert_close('dv', ref_dv, tri_dv, 0.007)
    # assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    # assert_close('dg', ref_dg, tri_dg, 0.015)
    # assert_close('dh0', ref_dh0, tri_dh0, 0.007)


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(
                *test
            ),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.float16),
            # bfloat16 tests
            (2, 500, 3, 60, 1, 1, 0, False, torch.bfloat16),
            (2, 1000, 3, 64, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.bfloat16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.bfloat16),
        ]
    ],
)
def test_chunk_opt(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(
            reason="chunk_gated_delta_rule_opt is not supported on alchemist for D>128"
        )

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(True) for x in (q, k, v, beta, g, h0)
    )

    tri, tri_ht = chunk_gated_delta_rule_opt(
        q=(
            F.normalize(q.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else q.clone()
        ),
        k=(
            F.normalize(k.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else k.clone()
        ),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="hip-B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(
                *test
            ),
        )
        for test in [
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.bfloat16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, torch.bfloat16),
            (1, 63, 1, 128, 1, 1, 0, False, torch.bfloat16),
            (2, 500, 3, 128, 1, 1, 0, False, torch.bfloat16),
        ]
    ],
)
@pytest.mark.parametrize(
    "state_dtype",
    [
        pytest.param(torch.float32, id="state_fp32"),
        pytest.param(torch.bfloat16, id="state_bf16"),
    ],
)
@pytest.mark.skipif(not IS_AMD, reason="Skipping HIP-only test on non-AMD backend")
@pytest.mark.skipif(
    _is_gfx12_runtime(),
    reason="chunk_gated_delta_rule_fwd_h_hip_fn kernel does not support gfx12!",
)
def test_chunk_opt_hip(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
):
    torch.manual_seed(42)
    if D != 128 or dtype != torch.bfloat16:
        pytest.skip(reason="HIP kernel requires D=128 and bfloat16")

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(True) for x in (q, k, v, beta, g, h0)
    )
    initial_state = h0.clone().to(state_dtype).transpose(-1, -2).contiguous()

    tri, tri_ht = chunk_gated_delta_rule_opt_vk(
        q=(
            F.normalize(q.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else q.clone()
        ),
        k=(
            F.normalize(k.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else k.clone()
        ),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_chunk_hip=True,
        state_dtype=state_dtype,
    )

    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    tol = 0.005 if state_dtype == torch.float32 else 0.02
    assert tri_ht.dtype == state_dtype
    assert_close("o", ref.float(), tri.float(), tol)
    assert_close("ht", ref_ht.float(), tri_ht.transpose(-1, -2).float(), tol)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
            # bfloat16 tests
            (4, 60, 0, [0, 15], torch.bfloat16),
            (4, 64, 0, [0, 256, 500, 1000], torch.bfloat16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_opt_varlen because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_opt_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(
            reason="chunk_gated_delta_rule_opt is not supported on alchemist for D>128"
        )
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(False) for x in (q, k, v, beta, g, h0)
    )

    tri, tri_ht = chunk_gated_delta_rule_opt(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
            q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="hip-H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 128, 0, [0, 15], torch.bfloat16),
            (4, 128, 0, [0, 256, 500, 1000], torch.bfloat16),
            (4, 128, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_opt_varlen_hip because SKIP_TEST_CHUNK_VARLEN is set",
)
@pytest.mark.parametrize(
    "state_dtype",
    [
        pytest.param(torch.float32, id="state_fp32"),
        pytest.param(torch.bfloat16, id="state_bf16"),
    ],
)
@pytest.mark.skipif(not IS_AMD, reason="Skipping HIP-only test on non-AMD backend")
@pytest.mark.skipif(
    _is_gfx12_runtime(),
    reason="chunk_gated_delta_rule_fwd_h_hip_fn kernel does not support gfx12!",
)
def test_chunk_opt_varlen_hip(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    state_dtype: torch.dtype,
):
    if D != 128 or dtype != torch.bfloat16:
        pytest.skip(reason="HIP kernel requires D=128 and bfloat16")
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(False) for x in (q, k, v, beta, g, h0)
    )
    initial_state = h0.clone().to(state_dtype).transpose(-1, -2).contiguous()

    tri, tri_ht = chunk_gated_delta_rule_opt_vk(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_chunk_hip=True,
        state_dtype=state_dtype,
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
            q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    tol = 0.005 if state_dtype == torch.float32 else 0.02
    assert tri_ht.dtype == state_dtype
    assert_close("o", ref.float(), tri.float(), tol)
    assert_close("ht", ref_ht.float(), tri_ht.transpose(-1, -2).float(), tol)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("triton", id="triton"),
        pytest.param(
            "hip",
            id="hip",
            marks=[
                pytest.mark.skipif(
                    not IS_AMD, reason="HIP backend requires an AMD device"
                ),
                pytest.mark.skipif(
                    _is_gfx12_runtime(),
                    reason="chunk_gated_delta_rule_fwd_h_hip_fn does not support gfx12!",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    [
        pytest.param(*test, id="indice-H{}-D{}-mask_p{}-cu_seqlens{}".format(*test))
        for test in [
            (4, 128, 0, [0, 15]),
            (4, 128, 0, [0, 256, 500, 1000]),
            (4, 128, 0.5, [0, 256, 500, 1000]),
            (4, 128, 0, [0, 15, 100, 300, 1200, 2000]),
        ]
    ],
)
@pytest.mark.parametrize(
    "state_dtype",
    [
        pytest.param(torch.float32, id="state_fp32"),
        pytest.param(torch.bfloat16, id="state_bf16"),
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_opt_vk_indice because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_opt_vk_indice(
    backend: str,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    state_dtype: torch.dtype,
):
    """Functional test for the indexed state-pool fwd_h on both backends.

    The Triton (``chunk_gated_delta_rule_fwd_h_opt_vk``) and HIP
    (``chunk_gated_delta_rule_fwd_h_hip_fn``) entries share the same dense/indexed
    contract: the only differences from the dense path are the per-sequence slot
    gather (``initial_state_indices``) on read and the in-place write-back into
    that slot. So when the pool holds the same initial states at (scattered)
    slots, dense vs indexed must be bit-identical, and non-indexed pool slots must
    stay untouched. The check is a same-backend self-comparison, so it does not
    depend on the gate layout being interpreted a particular way.
    """
    if backend == "hip":
        fwd_h = chunk_gated_delta_rule_fwd_h_hip_fn
        # HIP kernel is specialized for D=128 / bf16 and reads head-major gates.
        if D != 128:
            pytest.skip(reason="HIP kernel requires D=128 and bfloat16")
        extra_kwargs = {"g_head_major": True}
    else:
        fwd_h = chunk_gated_delta_rule_fwd_h_opt_vk
        extra_kwargs = {}

    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = int(cu_seqlens[-1])
    N = len(cu_seqlens) - 1
    B = 1

    # fwd_h-stage inputs: k [B,T,H,K] token-major, w/u [B,H,T,K|V] head-major,
    # g head-major [B,H,T]. Hg == H (no GQA).
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    u = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    g = F.logsigmoid(torch.rand(B, H, T, dtype=torch.float32, device=device))
    g = g * (torch.rand_like(g) > mask_p)

    # Dense per-sequence initial states [N, H, V, K] (random, so the gather is
    # actually exercised -- a wrong slot would read different values).
    h0 = torch.randn(N, H, D, D, dtype=state_dtype, device=device)

    # --- dense reference: slot == i_n ---
    h_ref, vnew_ref, ht_ref = fwd_h(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        **extra_kwargs,
    )

    # --- indexed pool: scatter the N states into a larger pool at unique,
    # non-identity slots to prove the gather honours initial_state_indices ---
    pool_size = N + 7
    perm = torch.randperm(pool_size, device=device)
    indices = perm[:N].to(torch.int32)
    pool = torch.randn(pool_size, H, D, D, dtype=state_dtype, device=device)
    pool_before = pool.clone()
    pool[indices.long()] = h0.clone()

    # Indexed pool path: passing initial_state_indices switches to slot gather +
    # in-place write-back (final_state aliases pool).
    h_idx, vnew_idx, ht_idx = fwd_h(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone(),
        initial_state=pool,
        initial_state_indices=indices,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        **extra_kwargs,
    )
    assert ht_idx is pool  # in-place: final state aliases the pool buffer

    # 1. snapshots + recomputed values are bit-identical to the dense path
    assert torch.equal(h_idx, h_ref), "h snapshots differ between dense and indexed"
    assert torch.equal(vnew_idx, vnew_ref), "v_new differs between dense and indexed"

    # 2. in-place write-back: the final state landed in the indexed pool slots
    # and equals the dense final state
    assert torch.equal(
        pool[indices.long()], ht_ref
    ), "in-place final state at indexed slots differs from dense final_state"

    # 3. non-indexed pool slots must be left exactly as they were
    untouched = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched[indices.long()] = False
    assert torch.equal(
        pool[untouched], pool_before[untouched]
    ), "non-indexed pool slots were modified by the kernel"


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(
                *test
            ),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, False, torch.bfloat16),
            (2, 1000, 3, 64, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.bfloat16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.bfloat16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.bfloat16),
        ]
    ],
)
def test_chunk_opt_vk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(
            reason="chunk_gated_delta_rule_opt_vk is not supported on alchemist for D>128"
        )

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(True) for x in (q, k, v, beta, g, h0)
    )

    # opt_vk expects initial_state in [N, H, V, K] layout
    tri, tri_ht = chunk_gated_delta_rule_opt_vk(
        q=(
            F.normalize(q.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else q.clone()
        ),
        k=(
            F.normalize(k.clone(), p=2, dim=-1)
            if not use_qk_l2norm_in_kernel
            else k.clone()
        ),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone().transpose(-1, -2).contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    assert_close("o", ref, tri, 0.005)
    # ref_ht is [B, H, K, V], tri_ht is [N, H, V, K]
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
            (4, 60, 0, [0, 15], torch.bfloat16),
            (4, 64, 0, [0, 256, 500, 1000], torch.bfloat16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_opt_vk_varlen because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_opt_vk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(
            reason="chunk_gated_delta_rule_opt_vk is not supported on alchemist for D>128"
        )
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = (
        x.to(device).requires_grad_(False) for x in (q, k, v, beta, g, h0)
    )

    # opt_vk expects initial_state in [N, H, V, K] layout
    tri, tri_ht = chunk_gated_delta_rule_opt_vk(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone().transpose(-1, -2).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
            q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    assert_close("o", ref, tri, 0.005)
    # ref_ht is [N, H, K, V], tri_ht is [N, H, V, K]
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)


# --- Fused FlyDSL K1..K4 prepare (use_prepare_flydsl) ---------------------
#
# The fused kernel replaces the Triton K1+K2 pair and reproduces its output
# layouts ([B, H, T, K/V] head-major w/u, [B, H, T] g_cumsum) and exponent
# domain exactly, so the tests below hold K5/K6 fixed and compare the flag on
# vs off. Checking against recurrent_gated_delta_rule_ref instead would bury
# the swap under the pipeline's own, much larger, algebraic error.

try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        gdn_prepare_flydsl_supported,
    )
    from aiter.ops.flydsl.utils import is_flydsl_available

    _HAS_FLYDSL_PREPARE = is_flydsl_available()
except ImportError:  # pragma: no cover - import guard
    _HAS_FLYDSL_PREPARE = False

requires_flydsl_prepare = pytest.mark.skipif(
    not _HAS_FLYDSL_PREPARE,
    reason="flydsl is not installed, so use_prepare_flydsl cannot be exercised",
)

# The swap only perturbs the last bits of the bf16 w/u handed to K5, but K5
# accumulates them across chunks and K6 mixes them into o, so the end-to-end gap
# is a few bf16 ULPs on values of order 1. Asserted on absolute error rather
# than via assert_close: the whole claim is bit-level agreement between two
# paths, and assert_close's relative ratio downgrades to a warning under
# FLA_CI_ENV, which would let a real divergence through.
_PREPARE_ATOL_O = 6e-3
_PREPARE_ATOL_HT = 1e-2

# The fused prepare emits one set of w/u/g_cumsum regardless of which K5 reads
# them, and each K5 backend's own correctness is covered by the tests above, so
# the backend sweep rides on the varlen case only -- the harder address path.
_PREPARE_K5_BACKENDS = [
    pytest.param({}, id="k5_triton_vk"),
    pytest.param({"use_chunk_flydsl": True}, id="k5_flydsl"),
    pytest.param(
        {"use_chunk_hip": True},
        id="k5_hip",
        marks=pytest.mark.skipif(
            not IS_AMD or _is_gfx12_runtime(),
            reason="HIP K5 kernel requires a non-gfx12 AMD device",
        ),
    ),
]


def _prepare_make_inputs(B, T, Hg, H, D, *, n_state, seed, dtype=torch.bfloat16):
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = {"dtype": dtype, "device": device, "generator": gen}
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device, generator=gen)
    # Decay gates are negative, as produced by the GDN gating stage.
    g = -(
        torch.rand(B, T, H, dtype=torch.float32, device=device, generator=gen) * 0.5
        + 0.2
    )
    return {
        "q": torch.randn(B, T, Hg, D, **kw) * 0.2,
        "k": torch.randn(B, T, Hg, D, **kw) * 0.2,
        "v": torch.randn(B, T, H, D, **kw) * 0.2,
        "beta": beta.sigmoid(),
        "g": g,
        "initial_state": torch.randn(
            n_state, H, D, D, dtype=torch.float32, device=device, generator=gen
        )
        * 0.1,
    }


def _run_prepare_pipeline(inputs, *, cu_seqlens, use_prepare_flydsl, **kwargs):
    return chunk_gated_delta_rule_opt_vk(
        **{name: t.clone() for name, t in inputs.items()},
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_prepare_flydsl=use_prepare_flydsl,
        **kwargs,
    )


def _assert_prepare_swap_is_transparent(inputs, *, cu_seqlens, **kwargs):
    o_fused, ht_fused = _run_prepare_pipeline(
        inputs, cu_seqlens=cu_seqlens, use_prepare_flydsl=True, **kwargs
    )
    o_triton, ht_triton = _run_prepare_pipeline(
        inputs, cu_seqlens=cu_seqlens, use_prepare_flydsl=False, **kwargs
    )

    assert o_fused.shape == o_triton.shape
    assert ht_fused.shape == ht_triton.shape
    err_o = (o_fused.float() - o_triton.float()).abs().max().item()
    err_ht = (ht_fused.float() - ht_triton.float()).abs().max().item()
    assert err_o < _PREPARE_ATOL_O, f"o diff {err_o:.3e}"
    assert err_ht < _PREPARE_ATOL_HT, f"ht diff {err_ht:.3e}"


@requires_flydsl_prepare
@pytest.mark.parametrize(
    ("B", "T", "Hg", "H", "seed"),
    [
        # B > 1 takes the dense head-major output base ((i_b*H + i_h)*T), a
        # different branch from varlen's (i_h*T + bos).
        pytest.param(2, 192, 4, 4, 1, id="dense_b2"),
        # Hg != H exercises the i_h // (H//Hg) key/value broadcast.
        pytest.param(1, 256, 4, 16, 2, id="dense_gqa"),
        # T % 64 != 0 exercises the ragged-tail store clamp, which is
        # per-(sequence, head) under the head-major epilogue: a sequence-level
        # limit would spill the tail rows into the NEXT head's slab instead of
        # out of the tensor.
        pytest.param(1, 300, 4, 8, 3, id="dense_ragged"),
    ],
)
def test_chunk_opt_vk_prepare_flydsl(B: int, T: int, Hg: int, H: int, seed: int):
    inputs = _prepare_make_inputs(B, T, Hg, H, 128, n_state=B, seed=seed)
    _assert_prepare_swap_is_transparent(inputs, cu_seqlens=None)


@requires_flydsl_prepare
@pytest.mark.parametrize("k5_kwargs", _PREPARE_K5_BACKENDS)
@pytest.mark.parametrize("with_metadata", [False, True], ids=["cached", "metadata"])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping varlen test because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_opt_vk_prepare_flydsl_varlen(with_metadata: bool, k5_kwargs: dict):
    """Skewed, GQA, ragged varlen batch against every K5 backend.

    The fused prepare launches a rectangular (chunk column, sequence x head)
    grid, so it needs the LONGEST sequence's chunk count -- not the flattened
    total the other stages use. Mixing one long sequence with short ragged ones
    is what catches that, and ``with_metadata`` covers both ways the count is
    derived: off the prebuilt schedule, or off the identity-keyed caches.
    """
    seq_lens = [1024, 15, 100, 300]
    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int64, device=device)
    inputs = _prepare_make_inputs(
        1, cu_list[-1], 4, 16, 128, n_state=len(seq_lens), seed=4
    )

    kwargs = dict(k5_kwargs)
    if with_metadata:
        kwargs["prefill_metadata"] = build_gated_delta_rule_prefill_metadata(
            seq_lens, cu_seqlens=cu_seqlens, chunk_size=64
        )

    _assert_prepare_swap_is_transparent(inputs, cu_seqlens=cu_seqlens, **kwargs)


@requires_flydsl_prepare
@pytest.mark.parametrize("with_metadata", [False, True], ids=["cached", "metadata"])
def test_chunk_opt_vk_prepare_flydsl_decode_prefix(with_metadata: bool):
    """Leading decode-only sequences must be rebased away, as Triton does.

    ``cu_seqlens`` keeps the decode prefix while the data tensors are pre-sliced
    past it, so the fused kernel has to rebase the offsets itself. Getting this
    wrong reads the wrong tokens without raising, and no other test in this file
    exercises the decode prefix.
    """
    seq_lens = [1, 1, 128, 300]
    num_decodes = 2
    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int64, device=device)
    num_decode_tokens = sum(seq_lens[:num_decodes])
    n_prefill = len(seq_lens) - num_decodes

    inputs = _prepare_make_inputs(
        1, cu_list[-1] - num_decode_tokens, 4, 8, 128, n_state=n_prefill, seed=5
    )

    kwargs = {
        "num_decodes": num_decodes,
        "num_decode_tokens": num_decode_tokens,
    }
    if with_metadata:
        kwargs["prefill_metadata"] = build_gated_delta_rule_prefill_metadata(
            seq_lens,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )

    _assert_prepare_swap_is_transparent(inputs, cu_seqlens=cu_seqlens, **kwargs)


@requires_flydsl_prepare
@pytest.mark.parametrize(
    ("dtype", "D"),
    [
        # fp16 is the dangerous one: the fused kernel emits bf16 w/u, and K5
        # takes h from k.dtype but v_new from u.dtype, so accepting fp16 would
        # feed K6 mixed precision instead of raising.
        pytest.param(torch.float16, 128, id="fp16"),
        # D != 128 would trip compile_gdn_prepare's hard assert.
        pytest.param(torch.bfloat16, 64, id="head_dim_64"),
    ],
)
def test_chunk_opt_vk_prepare_flydsl_falls_back(dtype: torch.dtype, D: int):
    """Outside the fused kernel's support the flag must be refused, not asserted."""
    inputs = _prepare_make_inputs(1, 192, 4, 4, D, n_state=1, seed=6, dtype=dtype)
    assert not gdn_prepare_flydsl_supported(inputs["k"], inputs["v"])

    o_flag, ht_flag = _run_prepare_pipeline(
        inputs, cu_seqlens=None, use_prepare_flydsl=True
    )
    o_ref, ht_ref = _run_prepare_pipeline(
        inputs, cu_seqlens=None, use_prepare_flydsl=False
    )

    # The flag was refused, so this has to be the very same Triton computation.
    assert torch.equal(o_flag, o_ref)
    assert torch.equal(ht_flag, ht_ref)


@requires_flydsl_prepare
def test_chunk_opt_vk_prepare_flydsl_does_not_sync():
    """A warmed-up varlen forward must not read anything back to the host.

    The wrapper originally derived its grid with ``.tolist()`` on the device
    offsets, which blocks once per layer per forward; the identity-keyed caches
    exist to remove exactly that. Only the cached path is pinned here -- with a
    prebuilt schedule every host value is host-resident by construction.
    """
    cu_list = [0, 128, 300, 600]
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int64, device=device)
    inputs = _prepare_make_inputs(
        1, cu_list[-1], 4, 8, 128, n_state=len(cu_list) - 1, seed=7
    )

    # Warm up: JIT compilation and the one-shot memoized offset reads may
    # synchronize; steady-state forwards may not.
    for _ in range(2):
        _run_prepare_pipeline(inputs, cu_seqlens=cu_seqlens, use_prepare_flydsl=True)
    torch.cuda.synchronize()

    prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _run_prepare_pipeline(inputs, cu_seqlens=cu_seqlens, use_prepare_flydsl=True)
    finally:
        torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.attention.kda_replayssm import replayssm_sigmoid_gating_delta_rule
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() not in ("gfx1250", "gfx1250-strict"), reason="gfx1250 only"
)

TP = 8
H = 96 // TP  # local KDA heads, HV == H
K = V = 128
PROJ = H * K
FUSED_IN = 4 * PROJ + H + K  # in_proj's fused output [q | k | v | g | b | f_a]: 6284
CAP = 16  # ATOM_REPLAYSSM_CACHE_LEN's default, at least twice the verify window
LOWER_BOUND = -5.0  # linear_attn_config.gate_lower_bound
DEVICE = "cuda"
T_MAX = {"spec_verify": 5, "decode": 1}  # replayssm_max_query_len: num_spec + 1


def kda_gate(a, A_log, dt_bias, lower_bound):
    """The per-K-channel log-decay for a: [..., HV, K]."""
    x = a.float() + dt_bias.float().view(-1, K)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(A_log.float().exp()[:, None] * x)
    return -A_log.float().exp()[:, None] * F.softplus(x, threshold=20.0)


def ref_replayssm_kda(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    max_query_len,
    scale=None,
    use_qk_l2norm=True,
    lower_bound=None,
):
    """Same arguments as the ATOM entry point; none is modified.

    Returns o [T_tot, HV, V] (rows of skipped requests are 0), the updated copies of ckpt, buf_k, buf_u and buf_g,
    and which of them it wrote: flushed [num_slots] and record [num_slots, CAP].
    """
    ckpt, buf_k, buf_u, buf_g = (t.clone() for t in (ckpt, buf_k, buf_u, buf_g))
    _, T_tot, Hq, Kd = q.shape
    HV, Vd = v.shape[2], v.shape[3]
    cap = buf_k.shape[2]
    scale = Kd**-0.5 if scale is None else scale
    o = torch.zeros(T_tot, HV, Vd, dtype=q.dtype, device=q.device)
    flushed = torch.zeros(ckpt.shape[0], dtype=torch.bool, device=q.device)
    record = torch.zeros(ckpt.shape[0], cap, dtype=torch.bool, device=q.device)
    head_of = torch.arange(HV, device=q.device) // (HV // Hq)
    for n in range(slot_idx.numel()):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        slot = int(slot_idx[n])
        if eos == bos or slot < 0:
            continue
        h = max(int(write_pos[slot]), 0)
        flush = h + 2 * max_query_len > cap
        base = 0 if flush else h
        # Rebuild the state from the checkpoint and the h committed records: [HV, V, K], V first
        S = ckpt[slot].float()
        for j in range(h):
            S = S * buf_g[slot, :, j].float().exp()[:, None, :]
            S = (
                S
                + buf_u[slot, :, j].float()[:, :, None]
                * buf_k[slot, :, j].float()[:, None, :]
            )
        if flush:
            ckpt[slot] = S.to(ckpt.dtype)
            flushed[slot] = True
        for t in range(eos - bos):
            i = bos + t
            qt, kt, vt = (
                q[0, i, head_of].float(),
                k[0, i, head_of].float(),
                v[0, i].float(),
            )
            g = kda_gate(a[0, i], A_log, dt_bias, lower_bound)  # [HV, K]
            beta = torch.sigmoid(b[0, i].float())  # [HV]
            if use_qk_l2norm:
                qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            qt = qt * scale
            S = S * g.exp()[:, None, :]
            u = (vt - (S @ kt[:, :, None]).squeeze(-1)) * beta[:, None]
            S = S + u[:, :, None] * kt[:, None, :]
            o[i] = (S @ qt[:, :, None]).squeeze(-1).to(o.dtype)
            pos = base + t
            buf_k[slot, :, pos] = kt.to(buf_k.dtype)
            buf_u[slot, :, pos] = u.to(buf_u.dtype)
            buf_g[slot, :, pos] = g.to(buf_g.dtype)
            record[slot, pos] = True
    return o, ckpt, buf_k, buf_u, buf_g, flushed, record


def make_inputs(m, mode, pad_rows=1, seed=0, *, t=None, cap=CAP, dtype=torch.bfloat16):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (
            torch.randn(*shape, generator=gen, device=DEVICE, dtype=torch.float32)
            * scale
        )

    bf16 = dtype
    T = T_MAX[mode] if t is None else t
    rows = m + pad_rows
    T_tot = rows * T
    num_slots = 2 * m + 3
    # causal_conv1d_update's q, k, v and f_b_proj's gate, each "t (h d) -> 1 t h d"
    q, k, v, gate = (randn(1, T_tot, H, K).to(bf16) for _ in range(4))
    # beta is a column slice of the fused in_proj output
    beta = randn(T_tot, FUSED_IN).to(bf16)[:, 4 * PROJ : 4 * PROJ + H].unsqueeze(0)
    A_log = (
        torch.empty(H, device=DEVICE, dtype=torch.float32)
        .uniform_(1.0, 16.0, generator=gen)
        .log()
    )
    dt_bias = randn(PROJ, scale=0.5)
    ckpt = randn(num_slots, H, V, K, scale=0.1)  # ssm_state: fp32, [slot, HV, V, K]
    # Records as the kernel writes them: L2-normalised k, u, and a log-decay g in (lower_bound, 0)
    buf_k = F.normalize(randn(num_slots, H, cap, K), dim=-1).to(bf16)
    buf_u = randn(num_slots, H, cap, V, scale=0.1).to(bf16)
    buf_g = LOWER_BOUND * torch.sigmoid(randn(num_slots, H, cap, K))
    slots = torch.randperm(num_slots, generator=gen, device=DEVICE)[:m]
    # The cursor stays within -1 (just after prefill) .. cap - T; past cap - 2T the forward flushes.
    write_pos = torch.randint(
        -1, cap - T + 1, (num_slots,), generator=gen, device=DEVICE, dtype=torch.int32
    )
    cursors = torch.tensor(
        [-1, 0, cap - 2 * T, cap - 2 * T + 1, cap - T], dtype=torch.int32, device=DEVICE
    )
    write_pos[slots] = cursors[torch.arange(m, device=DEVICE) % cursors.numel()]
    slot_idx = torch.cat(
        [slots.to(torch.int32), slots.new_full((pad_rows,), -1).to(torch.int32)]
    )
    cu_seqlens = torch.arange(0, (rows + 1) * T, T, dtype=torch.int32, device=DEVICE)
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "a": gate,
        "b": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "ckpt": ckpt,
        "buf_k": buf_k,
        "buf_u": buf_u,
        "buf_g": buf_g,
        "write_pos": write_pos,
        "slot_idx": slot_idx,
        "cu_seqlens": cu_seqlens,
        "max_query_len": T,
    }
    return inputs, m * T


# m reaches max-num-seqs, 128 at c128; decode is the off-recipe ATOM_ENABLE_REPLAYSSM=1 case
CASES = [("spec_verify", m) for m in (1, 8, 32, 128)] + [
    ("decode", m) for m in (1, 8, 32)
]


@pytest.mark.parametrize("mode,m", CASES)
def test_replayssm_kda_fwd(mode, m):
    inp, live = make_inputs(m, mode)
    ref_o, ref_ckpt, ref_bk, ref_bu, ref_bg, flushed, record = ref_replayssm_kda(
        **inp, use_qk_l2norm=True, lower_bound=LOWER_BOUND
    )

    state = {name: inp[name].clone() for name in ("ckpt", "buf_k", "buf_u", "buf_g")}
    write_pos = inp["write_pos"].clone()
    out = torch.zeros(inp["q"].shape[1], H, V, dtype=torch.bfloat16, device=DEVICE)
    replayssm_sigmoid_gating_delta_rule(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        ckpt=state["ckpt"],
        buf_k=state["buf_k"],
        buf_u=state["buf_u"],
        buf_g=state["buf_g"],
        write_pos=write_pos,
        slot_idx=inp["slot_idx"],
        cu_seqlens=inp["cu_seqlens"],
        max_query_len=inp["max_query_len"],
        o=out,
        use_qk_l2norm_in_kernel=True,
        lower_bound=LOWER_BOUND,
    )

    torch.testing.assert_close(out[:live], ref_o[:live], atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state["ckpt"], ref_ckpt, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state["buf_k"], ref_bk, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(state["buf_u"], ref_bu, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state["buf_g"], ref_bg, atol=1e-5, rtol=1e-5)
    # What the kernel does not write stays as it was; the cursor only moves in replayssm_commit
    assert torch.equal(state["ckpt"][~flushed], inp["ckpt"][~flushed])
    for name in ("buf_k", "buf_u", "buf_g"):
        assert torch.equal(
            state[name].transpose(1, 2)[~record], inp[name].transpose(1, 2)[~record]
        )
    assert torch.equal(write_pos, inp["write_pos"])
    assert torch.equal(out[live:], torch.zeros_like(out[live:]))


def _clone_state(inputs):
    return {
        **inputs,
        **{name: inputs[name].clone() for name in ("ckpt", "buf_k", "buf_u", "buf_g")},
    }


def _check_transition(inputs, out):
    expected = ref_replayssm_kda(**inputs, use_qk_l2norm=True, lower_bound=LOWER_BOUND)
    metadata = {
        key: inputs[key].clone() for key in ("write_pos", "slot_idx", "cu_seqlens")
    }
    result = replayssm_sigmoid_gating_delta_rule(
        **inputs, o=out, use_qk_l2norm_in_kernel=True, lower_bound=LOWER_BOUND
    )
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected[0], atol=2e-3, rtol=2e-2)
    for key, want, atol, rtol in zip(
        ("ckpt", "buf_k", "buf_u", "buf_g"),
        expected[1:5],
        (1e-5, 1e-3, 2e-3, 1e-5),
        (1e-4, 1e-2, 2e-2, 1e-5),
    ):
        torch.testing.assert_close(inputs[key], want, atol=atol, rtol=rtol)
    for key, want in metadata.items():
        assert torch.equal(inputs[key], want)


@pytest.mark.parametrize("live", [109, 128])
@pytest.mark.parametrize("seed", [0, 7])
def test_verify_two_transitions(live, seed):
    inputs, _ = make_inputs(128, "spec_verify", pad_rows=0, seed=seed)
    if live < 128:
        for key in ("q", "k", "v", "a", "b"):
            inputs[key] = inputs[key][:, : live * 5]
        inputs["slot_idx"][live:] = -1
        inputs["cu_seqlens"][live:] = live * 5
    out = torch.zeros((live * 5, H, V), device=DEVICE, dtype=inputs["v"].dtype)
    _check_transition(inputs, out)
    inputs["q"] = inputs["q"] * 0.8
    _check_transition(inputs, out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("t,cap", [(1, 16), (5, 16), (12, 32)])
def test_dtypes_and_chunks(dtype, t, cap):
    inputs, _ = make_inputs(
        8, "spec_verify", pad_rows=0, seed=11, t=t, cap=cap, dtype=dtype
    )
    out = torch.zeros((8 * t, H, V), device=DEVICE, dtype=dtype)
    _check_transition(inputs, out)
    _check_transition(inputs, out)


def test_capture_and_side_stream():
    inputs, _ = make_inputs(8, "spec_verify", pad_rows=0)
    captured = _clone_state(inputs)
    eager = _clone_state(inputs)
    out = torch.zeros((40, H, V), device=DEVICE, dtype=torch.bfloat16)
    eager_out = torch.zeros_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        replayssm_sigmoid_gating_delta_rule(
            **captured, o=out, use_qk_l2norm_in_kernel=True, lower_bound=LOWER_BOUND
        )
    torch.cuda.current_stream().wait_stream(stream)
    # Restore state after compiling. Capture records the call but does not execute it.
    for key in ("ckpt", "buf_k", "buf_u", "buf_g"):
        captured[key].copy_(inputs[key])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayssm_sigmoid_gating_delta_rule(
            **captured, o=out, use_qk_l2norm_in_kernel=True, lower_bound=LOWER_BOUND
        )
    graph.replay()
    replayssm_sigmoid_gating_delta_rule(
        **eager, o=eager_out, use_qk_l2norm_in_kernel=True, lower_bound=LOWER_BOUND
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager_out, atol=0, rtol=0)
    for key in ("ckpt", "buf_k", "buf_u", "buf_g"):
        torch.testing.assert_close(captured[key], eager[key], atol=0, rtol=0)


def test_reject_short_record_capacity():
    inputs, _ = make_inputs(1, "spec_verify", pad_rows=0)
    inputs["max_query_len"] = 9
    with pytest.raises(ValueError, match="capacity"):
        replayssm_sigmoid_gating_delta_rule(**inputs)


def test_slot_zero_output_allocation():
    inputs, _ = make_inputs(1, "spec_verify", pad_rows=0, seed=3)
    inputs["slot_idx"].fill_(0)
    expected = ref_replayssm_kda(**inputs, use_qk_l2norm=True, lower_bound=LOWER_BOUND)
    result = replayssm_sigmoid_gating_delta_rule(
        **inputs, use_qk_l2norm_in_kernel=True, lower_bound=LOWER_BOUND
    )
    assert result.shape == (1, 5, H, V)
    torch.testing.assert_close(result[0], expected[0], atol=2e-3, rtol=2e-2)


def test_all_skipped_requests_leave_buffers_unchanged():
    inputs, _ = make_inputs(8, "spec_verify", pad_rows=0)
    inputs["slot_idx"].fill_(-1)
    before = {
        key: value.clone()
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    out = torch.full((40, H, V), 7.0, device=DEVICE, dtype=torch.bfloat16)
    replayssm_sigmoid_gating_delta_rule(
        **inputs, o=out, use_qk_l2norm_in_kernel=True, lower_bound=LOWER_BOUND
    )
    assert torch.all(out == 7.0)
    for key, value in before.items():
        assert torch.equal(inputs[key], value), key

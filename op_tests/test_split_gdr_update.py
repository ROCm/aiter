# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)


def to_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    n, hv, k, v = state.shape
    assert k % 4 == 0, f"K ({k}) must be divisible by 4"
    return state.reshape(n, hv, k // 4, 4, v).permute(0, 1, 2, 4, 3).contiguous()


def from_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    n, hv, k4, v, four = state.shape
    assert four == 4, f"Last dimension must be 4, got {four}"
    return state.permute(0, 1, 2, 4, 3).reshape(n, hv, k4 * 4, v).contiguous()


def split_mixed_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
):
    bsz, _, seqlen = mixed_qkv.shape
    q_all = mixed_qkv[:, :key_dim, :]
    k_all = mixed_qkv[:, key_dim : 2 * key_dim, :]
    v_all = mixed_qkv[:, 2 * key_dim : 2 * key_dim + value_dim, :]
    q = q_all.transpose(1, 2).reshape(bsz, seqlen, num_heads_qk, head_dim).contiguous()
    k = k_all.transpose(1, 2).reshape(bsz, seqlen, num_heads_qk, head_dim).contiguous()
    v = v_all.transpose(1, 2).reshape(bsz, seqlen, num_heads_v, head_dim).contiguous()
    return q, k, v


def split_gdr_reference(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    bsz, _, seqlen = mixed_qkv.shape
    h = num_heads_qk
    hv = num_heads_v
    kdim = head_dim
    vdim = head_dim
    group_size = hv // h
    if scale is None:
        scale = kdim**-0.5

    mixed_qkv_f = mixed_qkv.float().cpu()
    A_log_f = A_log.float().cpu()
    dt_bias_f = dt_bias.float().cpu()
    a_f = a.float().cpu().view(bsz, seqlen, hv)
    b_f = b.float().cpu().view(bsz, seqlen, hv)

    state_f = torch.zeros(bsz, hv, kdim, vdim, dtype=torch.float32)
    idx_cpu = initial_state_indices.cpu()
    for n in range(bsz):
        idx = idx_cpu[n].item()
        if idx >= 0:
            state_f[n] = initial_state_source[idx].float().cpu()

    q_all = mixed_qkv_f[:, :key_dim, :]
    k_all = mixed_qkv_f[:, key_dim : 2 * key_dim, :]
    v_all = mixed_qkv_f[:, 2 * key_dim : 2 * key_dim + value_dim, :]
    output = torch.zeros(bsz, seqlen, hv, vdim, dtype=torch.float32)

    for t in range(seqlen):
        for i_hv in range(hv):
            i_h = i_hv // group_size
            q_vec = q_all[:, i_h * kdim : (i_h + 1) * kdim, t]
            k_vec = k_all[:, i_h * kdim : (i_h + 1) * kdim, t]
            v_vec = v_all[:, i_hv * vdim : (i_hv + 1) * vdim, t]

            a_t = a_f[:, t, i_hv]
            b_t = b_f[:, t, i_hv]
            x = a_t + dt_bias_f[i_hv]
            beta_x = softplus_beta * x
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                x,
            )
            g = -torch.exp(A_log_f[i_hv]) * softplus_x
            beta = torch.sigmoid(b_t)

            if use_qk_l2norm_in_kernel:
                q_vec = q_vec / torch.sqrt(torch.sum(q_vec * q_vec, dim=-1, keepdim=True) + 1e-6)
                k_vec = k_vec / torch.sqrt(torch.sum(k_vec * k_vec, dim=-1, keepdim=True) + 1e-6)

            q_vec = q_vec * scale
            state_f[:, i_hv, :, :] *= torch.exp(g).unsqueeze(-1).unsqueeze(-1)
            v_vec = v_vec - torch.einsum("bkv,bk->bv", state_f[:, i_hv, :, :], k_vec)
            v_vec = v_vec * beta.unsqueeze(-1)
            state_f[:, i_hv, :, :] += torch.einsum("bk,bv->bkv", k_vec, v_vec)
            output[:, t, i_hv, :] = torch.einsum("bkv,bk->bv", state_f[:, i_hv, :, :], q_vec)

    for n in range(bsz):
        idx = idx_cpu[n].item()
        if idx >= 0:
            initial_state_source[idx] = state_f[n].to(initial_state_source.dtype).to(initial_state_source.device)

    return output.to(mixed_qkv.dtype).to(mixed_qkv.device)


class TestFusedSplitGdrHipAiter:
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    def create_inputs(
        self,
        batch_size: int,
        seqlen: int,
        num_heads_qk: int,
        num_heads_v: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
        a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        ssm_state = torch.randn(
            batch_size + 10,
            num_heads_v,
            head_dim,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        return {
            "mixed_qkv": mixed_qkv,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_correctness(
        self, batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
    ):
        torch.manual_seed(42)
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim**-0.5

        ssm_state_ref = inputs["ssm_state"].clone()
        output_ref = split_gdr_reference(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        ssm_state_swizzled = to_swizzled_layout(inputs["ssm_state"].clone())
        output_hip = aiter.fused_split_gdr_update(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=ssm_state_swizzled,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

        output_diff = (output_ref - output_hip).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()
        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_all_kernels_performance(
        self, batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
    ):
        torch.manual_seed(42)
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim**-0.5

        num_warmup = 10
        num_iters = 1000

        q, k, v = split_mixed_qkv(
            inputs["mixed_qkv"],
            key_dim,
            value_dim,
            num_heads_qk,
            num_heads_v,
            head_dim,
        )
        a_3d = inputs["a"].view(batch_size, seqlen, num_heads_v).contiguous()
        b_3d = inputs["b"].view(batch_size, seqlen, num_heads_v).contiguous()

        ssm_state_std_template = inputs["ssm_state"].clone()
        ssm_state_swizzled_template = to_swizzled_layout(inputs["ssm_state"].clone())

        def _benchmark(run_fn, template):
            for _ in range(num_warmup):
                run_fn(template.clone())
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(num_iters):
                run_fn(template.clone())
            end_evt.record()
            torch.cuda.synchronize()
            return start_evt.elapsed_time(end_evt) / num_iters * 1000

        triton_us = _benchmark(
            lambda st: fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=a_3d,
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=q,
                k=k,
                v=v,
                b=b_3d,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            ),
            ssm_state_std_template,
        )

        hip_us = _benchmark(
            lambda st: aiter.fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b_gate=inputs["b"],
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            ),
            ssm_state_swizzled_template,
        )

        print(f"\n{'=' * 70}")
        print("Split GDR Kernels Performance Comparison (aiter)")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"  warmup={num_warmup}, iters={num_iters}")
        print(f"{'=' * 70}")
        print(f"  {'Kernel':<24s} {'Time (us)':>10s} {'vs Triton':>10s}")
        print(f"  {'-' * 24} {'-' * 10} {'-' * 10}")
        print(f"  {'Triton decode':<24s} {triton_us:10.2f} {'1.000x':>10s}")
        print(f"  {'HIP fused_split_gdr':<24s} {hip_us:10.2f} {triton_us / hip_us:9.3f}x")
        print(f"{'=' * 70}")

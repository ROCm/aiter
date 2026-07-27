# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import rope_cached_positions_fwd_inplace
from aiter.ops.triton.fusions.fused_reduce_qk_norm_rope_swa_write import (
    fused_reduce_qk_norm_rope_swa_write,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
    run_torch as run_torch_gemm_a8w8_blockscale,
)


def _build_cos_sin(
    rope_dim: int, max_seq: int, dtype: torch.dtype, device: torch.device
):
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(dtype).unsqueeze(-2).unsqueeze(-2)
    sin = torch.sin(freqs).to(dtype).unsqueeze(-2).unsqueeze(-2)
    return cos, sin


def run_torch(
    q_in_for_ref: torch.Tensor,
    kv_pre: torch.Tensor,
    q_norm_weight,
    kv_norm_weight,
    q_rms_eps: float,
    kv_rms_eps: float,
    rope_head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
    M: int,
    num_local_heads: int,
    head_dim: int,
    write_indices,
    batch_id_per_token,
    state_slot_per_seq,
    swa_kv_ref: torch.Tensor,
    win: int,
):
    """Reference: split-K reduce -> per-head Q RMSNorm + RoPE tail; per-row
    KV RMSNorm over head_dim + RoPE tail; optional SWA scatter."""
    dtype = kv_pre.dtype

    def _rms_per_row(x: torch.Tensor, w, eps: float) -> torch.Tensor:
        v = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + eps) * (w if w is not None else 1)).to(
            x.dtype
        )

    def _rope_inplace_slice(x_tail: torch.Tensor):
        rotate_style = 0 if is_neox else 1
        rope_cached_positions_fwd_inplace(
            x_tail,
            cos,
            sin,
            positions.view(1, -1),
            rotate_style=rotate_style,
            reuse_freqs_front_part=True,
            nope_first=False,
        )

    # Q: RMSNorm per head + RoPE on tail.
    q_ref = q_in_for_ref.to(dtype).view(M, num_local_heads, head_dim)
    q_ref = _rms_per_row(q_ref, q_norm_weight, q_rms_eps)
    q_tail = (
        q_ref[..., -rope_head_dim:]
        .contiguous()
        .view(1, M, num_local_heads, rope_head_dim)
    )
    _rope_inplace_slice(q_tail)
    q_ref[..., -rope_head_dim:] = q_tail.view(M, num_local_heads, rope_head_dim)

    # KV: RMSNorm over head_dim + RoPE on tail.
    kv_ref = kv_pre.clone()
    kv_ref = _rms_per_row(kv_ref, kv_norm_weight, kv_rms_eps)
    kv_tail = kv_ref[..., -rope_head_dim:].contiguous().view(1, M, 1, rope_head_dim)
    _rope_inplace_slice(kv_tail)
    kv_ref[..., -rope_head_dim:] = kv_tail.view(M, rope_head_dim).clone()

    # Optional SWA scatter using the *normed* kv values.
    slots = ring_idx = None
    if write_indices is not None:
        keep = write_indices >= 0
        src_ids = write_indices[keep].long()
        if src_ids.numel() > 0:
            src_kv = kv_ref[src_ids]
            src_pos = positions[src_ids]
            bids = batch_id_per_token[src_ids].long()
            slots = state_slot_per_seq[bids].long()
            ring_idx = src_pos % win
            swa_kv_ref[slots, ring_idx] = src_kv

    return q_ref, kv_ref, slots, ring_idx


@pytest.mark.parametrize("M", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("num_local_heads,q_lora_rank", [(8, 1024), (128, 1536)])
@pytest.mark.parametrize("rope_head_dim", [64])
@pytest.mark.parametrize("head_dim", [512])
@pytest.mark.parametrize("q_norm_eps", [1e-6])
@pytest.mark.parametrize("kv_norm_eps", [1e-6])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("with_swa", [True])
@pytest.mark.parametrize("num_splitk", [1, 2])
def test_fused_reduce_qk_norm_rope_swa_write(
    M: int,
    num_local_heads: int,
    q_lora_rank: int,
    rope_head_dim: int,
    head_dim: int,
    q_norm_eps: float,
    kv_norm_eps: float,
    is_neox: bool,
    with_swa: bool,
    num_splitk: int,
):
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    N = num_local_heads * head_dim
    K = q_lora_rank
    dtype = torch.bfloat16
    device = torch.device("cuda")

    x, w, _w_shfl, x_scale, _x_scale_shfl, w_scale, _ = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N,
            K,
            128,
            128,
            dtype=dtype,
            layout="TN",
            output=False,
            shuffle=True,
        )
    )

    kv_pre = torch.randn(M, head_dim, dtype=dtype, device=device)
    q_weight = None  # weight-free RMSNorm for Q
    kv_weight = None  # weight-free RMSNorm for KV

    # Compute the reference full GEMM output (post a8w8 blockscale dequant).
    y_gemm = run_torch_gemm_a8w8_blockscale(x, w, x_scale, w_scale, torch.float32).to(
        dtype
    )

    # Build q_in: 2D [M, N] when num_splitk==1, otherwise 3D [num_splitk, M, N]
    # whose split-K sum equals y_gemm. Reference q_ref uses the same fp32 sum
    # the kernel will see, so any bf16 partial-sum noise affects both equally.
    if num_splitk == 1:
        q_in = y_gemm  # 2D
        q_in_for_ref = y_gemm.float()
    else:
        partials = (
            torch.randn(num_splitk - 1, M, N, dtype=torch.float32, device=device) * 0.1
        )
        last = y_gemm.float() - partials.sum(dim=0)
        q_in_fp32 = torch.cat(
            [partials, last.unsqueeze(0)], dim=0
        )  # [num_splitk, M, N] fp32
        q_in = q_in_fp32.to(dtype)  # bf16 partials (what GEMM split-K returns)
        q_in_for_ref = q_in.float().sum(
            dim=0
        )  # what the kernel reads (bf16) summed in fp32

    max_seq = 8192
    cos, sin = _build_cos_sin(rope_head_dim, max_seq, dtype, device)
    positions = torch.randperm(max_seq, dtype=torch.int64, device=device)[:M]

    num_slots = M
    win = 128
    swa_kv_test = torch.zeros(num_slots, win, head_dim, dtype=dtype, device=device)
    swa_kv_ref = torch.zeros_like(swa_kv_test)

    write_indices = batch_id = slot_map = None
    if with_swa:
        write_indices = torch.randperm(M, dtype=torch.int32, device=device)
        slot_map = torch.randperm(num_slots, dtype=torch.int32, device=device)[:M]
        batch_id = torch.randperm(M, dtype=torch.int32, device=device)

    q_ref, kv_ref, slots, ring_idx = run_torch(
        q_in_for_ref,
        kv_pre,
        q_weight,
        kv_weight,
        q_norm_eps,
        kv_norm_eps,
        rope_head_dim,
        cos,
        sin,
        positions,
        is_neox,
        M,
        num_local_heads,
        head_dim,
        write_indices,
        batch_id,
        slot_map,
        swa_kv_ref,
        win,
    )

    kv_test = kv_pre.clone()
    q_out = torch.empty(M, num_local_heads, head_dim, dtype=dtype, device=device)

    fused_reduce_qk_norm_rope_swa_write(
        q_in,
        kv_test,
        q_weight,
        kv_weight,
        q_norm_eps,
        kv_norm_eps,
        rope_head_dim,
        cos,
        sin,
        positions,
        q_out=q_out,
        is_neox=is_neox,
        write_indices=write_indices,
        batch_id_per_token=batch_id,
        state_slot_mapping=slot_map,
        swa_kv=swa_kv_test,
        win=win,
        dtype=dtype,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=0.05, atol=0.1)
    torch.testing.assert_close(kv_test, kv_ref, rtol=0.05, atol=0.1)
    if with_swa:
        torch.testing.assert_close(
            swa_kv_test[slots, ring_idx],
            swa_kv_ref[slots, ring_idx],
            rtol=0.05,
            atol=0.1,
        )

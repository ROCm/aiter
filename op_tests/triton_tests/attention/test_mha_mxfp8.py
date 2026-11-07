# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.attention.mha_mxfp8 import mha_fwd
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp

arch = get_arch()


# ---------------
# reference
# ---------------


def mha_torch(q, k, v, sm_scale, causal, window_left=0, sink=None):
    # GQA: broadcast each K/V head to the group of query heads that shares it
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    qk = q @ k.transpose(-2, -1)
    qk = qk * sm_scale
    if causal:
        # bottom-right aligned: query i attends to keys j <= i + (SK - SQ), and
        # with a sliding window to keys j >= i + (SK - SQ) - window_left
        sq, sk = q.shape[2], k.shape[2]
        M = torch.tril(torch.ones((sq, sk), device=q.device), diagonal=sk - sq)
        if window_left > 0:
            M = torch.triu(M, diagonal=sk - sq - window_left)
        qk[:, :, M == 0] = float("-inf")
    m = qk.amax(-1, keepdim=True)
    if sink is not None:
        # the sink is an extra softmax column without a value vector
        sink = sink[None, :, None, None]
        m = torch.maximum(m, sink)
    # rows without a visible key (causal, SQ > SK) give zero
    p = torch.exp(qk - m)
    l = p.sum(-1, keepdim=True)
    if sink is not None:
        l = l + torch.exp(sink - m)
    p_q = p.to(torch.float8_e4m3fn).to(torch.float32)
    o = (p_q @ v) / l
    o = torch.where(torch.isinf(m), 0.0, o)
    return o


# ---------------
# initialize data
# ---------------


def init_qkv(shape, quant_dim, device="cuda"):
    x = torch.randn(shape, device=device, dtype=torch.float32) * 0.5
    x_quant, x_scale = downcast_to_mxfp(
        x.to(torch.bfloat16), torch.float8_e4m3fn, axis=quant_dim
    )
    x = upcast_from_mxfp(x_quant, x_scale, torch.bfloat16, axis=quant_dim).float()
    if quant_dim != -1:
        x_quant = x_quant.contiguous()
        x_scale = x_scale.contiguous()
    return x, x_quant, x_scale


# ---------------
# unit tests
# ---------------


@dataclass
class Case:
    b: int
    hq: int
    hk: int
    sq: int
    sk: int
    d: int
    # left sliding window size; 0 means no window
    sliding_window: int = 0
    # one attention sink logit for each query head
    sink: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # one K/V tile, one M tile: the shortest path through the pipeline
            Case(1, 2, 2, 64, 64, 128),
            Case(1, 1, 1, 128, 128, 128),
            Case(1, 8, 8, 256, 256, 128),
            Case(2, 8, 8, 512, 512, 128),
            Case(2, 8, 8, 512, 512, 64),
            Case(4, 32, 32, 1024, 1024, 128),
            Case(2, 16, 16, 2048, 2048, 128),
            Case(1, 8, 8, 4096, 4096, 128),
            Case(3, 5, 5, 1024, 1024, 128),
            # M not a multiple of BLOCK_M: the last program writes a partial tile
            Case(2, 4, 4, 600, 600, 128),
            # HEAD_DIM != HEAD_DIM_POW2: the descriptors pad the head dim to 128
            Case(2, 8, 8, 512, 512, 96),
            # GQA: several query heads share one K/V head
            Case(2, 8, 2, 512, 512, 128),
            Case(4, 32, 8, 1024, 1024, 128),
            Case(2, 16, 4, 1024, 1024, 64),
            Case(1, 48, 8, 2048, 2048, 128),
            # seqlen_q < seqlen_k
            Case(2, 8, 2, 512, 1024, 128),
            Case(2, 4, 4, 600, 700, 128),
            # seqlen_q > seqlen_k
            Case(1, 8, 8, 2048, 512, 128),
            Case(2, 8, 2, 1024, 300, 128),
            # sliding window: windows below, at and above BLOCK_N, partial M
            # tiles, GQA, SQ != SK, and a window that covers the sequence
            Case(1, 2, 2, 128, 128, 128, 63),
            Case(2, 8, 8, 1024, 1024, 128, 255),
            Case(1, 8, 8, 4096, 4096, 128, 1023),
            Case(2, 8, 2, 600, 600, 128, 300),
            # attention sink: also with GQA, SQ != SK, HEAD_DIM != HEAD_DIM_POW2
            # and a sliding window
            Case(2, 8, 8, 1024, 1024, 128, sink=True),
            Case(2, 8, 2, 600, 700, 128, sink=True),
            Case(1, 8, 8, 2048, 512, 128, 127, sink=True),
            Case(2, 8, 8, 1024, 1024, 128, 255, sink=True),
        ]
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_mha_mxfp8(b, hq, hk, sq, sk, d, sliding_window, sink, causal, device="cuda"):
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")
    if sliding_window > 0 and not causal:
        pytest.skip("a sliding window requires causal=True")
    torch.manual_seed(0)

    q_shape = (b, hq, sq, d)
    kv_shape = (b, hk, sk, d)
    q_ref, q_tri, q_scale = init_qkv(q_shape, -1, device)
    k_ref, k_tri, k_scale = init_qkv(kv_shape, -1, device)
    v_ref, v_tri, v_scale = init_qkv(kv_shape, -2, device)
    sm_scale = 1.0 / (d**0.5)

    sink_logits = (
        torch.randn((hq,), dtype=torch.float32, device=device) * 3 if sink else None
    )

    o_ref = mha_torch(
        q_ref,
        k_ref,
        v_ref,
        sm_scale,
        causal=causal,
        window_left=sliding_window,
        sink=sink_logits,
    )

    out = torch.empty(q_tri.shape, device=device, dtype=torch.bfloat16)
    o_tri = mha_fwd(
        q_tri,
        k_tri,
        v_tri,
        q_scale,
        k_scale,
        v_scale,
        o=out,
        sm_scale=sm_scale,
        causal=causal,
        window_size=(sliding_window, -1),
        sink=sink_logits,
    )
    o_tri = o_tri.float()

    torch.testing.assert_close(o_tri, o_ref, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "batch, hq, hk, d, max_seqlen_q, max_seqlen_k, sliding_window, sink",
    [
        (4, 8, 8, 128, 128, 128, 0, False),
        (8, 8, 2, 128, 512, 512, 0, False),
        (4, 16, 4, 64, 1024, 1024, 0, False),
        (3, 4, 4, 128, 1000, 700, 0, False),
        # many short sequences
        (16, 8, 8, 128, 300, 300, 0, False),
        # SQ mostly > SK, and SQ mostly < SK
        (6, 8, 8, 128, 2048, 256, 0, False),
        (6, 8, 8, 128, 256, 2048, 0, False),
        (2, 32, 8, 128, 4096, 4096, 0, False),
        # sliding window: unaligned starts, short sequences, SQ != SK
        (4, 8, 8, 128, 1000, 700, 63, False),
        (16, 8, 2, 128, 300, 300, 127, False),
        (6, 8, 8, 128, 256, 2048, 500, False),
        (2, 32, 8, 128, 4096, 4096, 1023, False),
        # attention sink, also with a sliding window
        (4, 8, 8, 128, 1000, 700, 0, True),
        (16, 8, 2, 128, 300, 300, 127, True),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_mha_mxfp8_varlen(
    batch,
    hq,
    hk,
    d,
    max_seqlen_q,
    max_seqlen_k,
    sliding_window,
    sink,
    causal,
    device="cuda",
):
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")
    if sliding_window > 0 and not causal:
        pytest.skip("a sliding window requires causal=True")
    torch.manual_seed(0)
    # Independent lengths in [1, max]: short sequences, unaligned starts,
    # and SQ > SK and SQ < SK in one batch.
    seqlens_q = torch.randint(
        1, max_seqlen_q + 1, (batch,), dtype=torch.int32, device=device
    )
    seqlens_k = torch.randint(
        1, max_seqlen_k + 1, (batch,), dtype=torch.int32, device=device
    )
    total_q, total_k = sum(seqlens_q), sum(seqlens_k)
    q_ref, q_tri, q_scale = init_qkv((hq, total_q, d), -1, device)
    k_ref, k_tri, k_scale = init_qkv((hk, total_k, d), -1, device)
    # V scales are quantized along the packed total_k axis
    v_ref, v_tri, v_scale = init_qkv((hk, total_k, d), -2, device)
    sm_scale = 1.0 / (d**0.5)
    sink_logits = (
        torch.randn((hq,), dtype=torch.float32, device=device) * 3 if sink else None
    )

    cu_seqlens_q = F.pad(seqlens_q.cumsum(0, dtype=torch.int32), (1, 0))
    cu_seqlens_k = F.pad(seqlens_k.cumsum(0, dtype=torch.int32), (1, 0))
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()

    o_ref = torch.zeros((hq, total_q, d), device=device, dtype=torch.float32)
    for i in range(seqlens_q.shape[0]):
        q0, q1, k0, k1 = cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]
        o_ref[:, q0:q1] = mha_torch(
            q_ref[None, :, q0:q1],
            k_ref[None, :, k0:k1],
            v_ref[None, :, k0:k1],
            sm_scale,
            causal=causal,
            window_left=sliding_window,
            sink=sink_logits,
        )[0]

    o_tri = mha_fwd(
        q_tri,
        k_tri,
        v_tri,
        q_scale,
        k_scale,
        v_scale,
        sm_scale=sm_scale,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q).item(),
        max_seqlen_k=max(seqlens_k).item(),
        window_size=(sliding_window, -1),
        sink=sink_logits,
    )
    o_tri = o_tri.float()

    torch.testing.assert_close(o_tri, o_ref, atol=0.02, rtol=0.02)

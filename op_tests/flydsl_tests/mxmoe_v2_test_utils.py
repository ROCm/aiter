# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class StandaloneResult:
    out: torch.Tensor
    ref: torch.Tensor
    payloads: dict[str, torch.Tensor]
    scales: dict[str, torch.Tensor]
    limit: float | None = None
    launch: Callable[[], torch.Tensor] | None = None
    # What the output would be if the K-pad tail were multiplied in instead of
    # read back as zero; only set when inter_dim_pad > 0.
    ref_if_pad_consumed: torch.Tensor | None = None

    @property
    def a_scale(self):
        return self.scales["a"]

    @property
    def b_scale(self):
        return self.scales["b"]


def logits_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x64 = x.double()
    y64 = y.double()
    denominator = x64.square().sum() + y64.square().sum() + 1e-8
    return float(1 - 2 * (x64 * y64).sum() / denominator)


def _poison_high_kpack(scale, groups, chunks):
    words = scale.view(torch.uint8).view(-1, 4)
    words_per_group = chunks * 64
    for group in range(groups):
        last = group * words_per_group + (chunks - 1) * 64
        words[last : last + 64, 2:4] = 0xFF


def run_standalone_v2_a8w8(
    *,
    BM,
    BN,
    BK,
    K,
    limit,
    poison_scale_padding=False,
    corrupt_b_scale_roll=0,
    poison_real_scale=False,
    epilog="atomic",
    use_nt=False,
    inter_dim_pad=0,
):
    from aiter import dtypes
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
    from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
        _mxfp4_a_scale_sorted_shuffled,
    )
    from aiter.ops.shuffle import shuffle_weight_a16w4
    from aiter.utility import fp4_utils

    device = "cuda"
    K_real = K - inter_dim_pad
    if K_real <= 0 or K_real % 32 != 0:
        raise AssertionError(
            f"inter_dim_pad must leave a 32-aligned real K; got K={K}, "
            f"inter_dim_pad={inter_dim_pad}"
        )
    E = topk = 1
    N = BN
    max_sorted = BM
    sti = torch.arange(BM, dtype=torch.int32, device=device)
    sei = torch.zeros(1, dtype=torch.int32, device=device)
    cumsum = torch.tensor([BM, BM], dtype=torch.int32, device=device)
    weights = torch.ones(BM, dtype=torch.float32, device=device)

    a_src = ((torch.arange(BM * K, device=device).view(BM, K) % 13) - 6).float() / 2
    b_src = ((torch.arange(N * K, device=device).view(N, K) * 3 % 17) - 8).float() / 2
    a_q = a_src.to(dtypes.fp8)
    b_q = b_src.to(dtypes.fp8).view(E, N, K)

    base = (torch.arange(K // 32, dtype=torch.uint8, device=device) % 8) * 2 + 124
    a_scale_raw = torch.stack([base.roll(i % base.numel()) for i in range(BM)]).view(
        dtypes.fp8_e8m0
    )
    b_scale_raw = torch.stack(
        [base.roll((i * 3) % base.numel()) for i in range(N)]
    ).view(dtypes.fp8_e8m0)

    a_scale = _mxfp4_a_scale_sorted_shuffled(
        a_scale_raw.view(torch.uint8),
        sti,
        cumsum,
        max_sorted,
        K,
        BM=BM,
        BK=256,
    )
    b_scale_for_kernel = (
        b_scale_raw.view(torch.uint8)
        .roll(corrupt_b_scale_roll, dims=1)
        .view(dtypes.fp8_e8m0)
        if corrupt_b_scale_roll
        else b_scale_raw
    )
    b_scale = fp4_utils.e8m0_shuffle(b_scale_for_kernel).view(torch.uint8)
    b_shuffled = shuffle_weight_a16w4(b_q, 16, False).view(torch.uint8)

    if poison_scale_padding:
        if K % 256 != 128:
            raise AssertionError("scale padding poison requires K % 256 == 128")
        chunks = (K + 255) // 256
        _poison_high_kpack(a_scale, max(1, BM // 32), chunks)
        _poison_high_kpack(b_scale, N // 32, chunks)
    if poison_real_scale:
        b_scale.view(-1)[0] = 0xFF

    out = torch.zeros((BM, N), dtype=torch.bfloat16, device=device)
    target = (
        torch.empty((BM, topk, N), dtype=torch.bfloat16, device=device)
        if epilog == "reduce"
        else out
    )

    def launch():
        mxfp4_moe_gemm2(
            inter_sorted_quant=a_q.view(torch.uint8),
            inter_sorted_shuffled_scale=a_scale,
            w2_u8=b_shuffled,
            w2_scale_u8=b_scale,
            sorted_expert_ids=sei,
            cumsum_tensor=cumsum,
            sorted_token_ids=sti,
            sorted_weights=weights,
            out=target,
            M_logical=BM,
            max_sorted=max_sorted,
            NE=E,
            D_HIDDEN=N,
            D_INTER=K,
            topk=topk,
            BM=BM,
            BN=BN,
            BK=BK,
            a_dtype="fp8",
            b_dtype="fp8",
            epilog=epilog,
            use_nt=use_nt,
            inter_dim_pad=inter_dim_pad,
        )
        if epilog == "reduce":
            from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

            _run_moe_reduction(target, out, BM, topk, N)
        return out

    launch()
    torch.cuda.synchronize()

    a_deq = a_q.float() * fp4_utils.e8m0_to_f32(a_scale_raw).repeat_interleave(32, 1)
    b_deq = b_q[0].float() * fp4_utils.e8m0_to_f32(b_scale_raw).repeat_interleave(32, 1)

    def _matmul(k):
        return (
            (a_deq[:, :k].double() @ b_deq[:, :k].double().T).to(torch.bfloat16).float()
        )

    return StandaloneResult(
        out=out.float(),
        ref=_matmul(K_real),
        payloads={"a": a_q, "b": b_shuffled},
        scales={"a": a_scale, "b": b_scale},
        limit=limit,
        launch=launch,
        ref_if_pad_consumed=_matmul(K) if inter_dim_pad else None,
    )

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import topk as topk_ops


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return get_gfx() in ("gfx1250", "gfx1250-strict")


pytestmark = pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250")


def _torch_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    need_renorm: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the legacy wave32 tie priority with ordinary torch operations."""
    rows, experts = logits.shape
    scores = torch.sigmoid(logits.float())
    choice = scores + bias.float()
    choice = torch.where(torch.isnan(choice), float("-inf"), choice)

    cols = torch.arange(experts, device=logits.device, dtype=torch.int64)
    lane = (cols // 4) % 32
    local_pos = (cols // 128) * 4 + cols % 4
    low2 = (lane & 3) ^ (((lane & 4) != 0).to(torch.int64) * 3)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((lane ^ 0x38) & 0x38) | bit2 | low2
    tie_rank = lane_rank * 32 + local_pos
    tie_rank = tie_rank.expand(rows, -1)

    selected_ids = []
    selected_weights = []
    for _ in range(topk):
        max_choice = choice.max(dim=1, keepdim=True).values
        ranks = torch.where(
            choice == max_choice,
            tie_rank,
            torch.iinfo(torch.int64).max,
        )
        ids = ranks.argmin(dim=1)
        selected_ids.append(ids)
        selected_weights.append(scores.gather(1, ids[:, None]).squeeze(1))
        choice.scatter_(1, ids[:, None], float("-inf"))

    ids = torch.stack(selected_ids, dim=1).to(torch.int32)
    weights = torch.stack(selected_weights, dim=1)
    if need_renorm:
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights *= torch.where(weight_sum > 0.0, scale / weight_sum, 0.0)
    else:
        weights *= scale
    return weights, ids


@pytest.mark.parametrize(
    "experts,topk,rows,case,need_renorm,scale",
    [
        (512, 4, 1, "random", True, 1.0),
        (768, 8, 8, "ties", True, 1.0),
        (768, 8, 2, "small_scores", True, 1.0),
        (896, 16, 32, "random", False, 2.5),
        (900, 8, 8, "nan", True, 1.0),
        (1024, 16, 128, "extreme", False, 0.75),
    ],
)
def test_exact_sigmoid_topk_matches_torch(
    experts, topk, rows, case, need_renorm, scale
):
    torch.manual_seed(experts + topk + rows)
    backing = torch.randn(
        (rows, experts + 17),
        dtype=torch.bfloat16,
        device="cuda",
    )
    logits = backing[:, 7 : 7 + experts]
    bias = (torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )

    if case == "ties":
        logits.zero_()
        bias.zero_()
    elif case == "small_scores":
        logits[0].fill_(-100)
        logits[1].fill_(-50)
        bias.zero_()
    elif case == "nan":
        logits[:, ::113] = float("nan")
        bias[::127] = float("nan")
    elif case == "extreme":
        logits[:, 0::4] = 100
        logits[:, 1::4] = -100
        logits[:, 2::4] = 0
        logits[:, 3::4] = torch.linspace(
            -20,
            20,
            experts // 4,
            device="cuda",
            dtype=torch.float32,
        )

    expected_weights, expected_ids = _torch_reference(
        logits,
        bias,
        topk,
        need_renorm,
        scale,
    )
    weights = torch.empty_strided(
        (rows, topk),
        (topk + 3, 1),
        dtype=torch.float32,
        device="cuda",
    )
    ids = torch.empty_strided(
        (rows, topk),
        (topk + 3, 1),
        dtype=torch.int32,
        device="cuda",
    )

    # A pitch of experts + 17 keeps biased_grouped_topk on the Triton kernel.
    assert topk_ops._can_use_gfx1250_exact_sigmoid_topk(logits, bias, weights, ids)
    topk_ops.biased_grouped_topk(
        logits,
        bias,
        weights,
        ids,
        1,
        1,
        need_renorm,
        scale,
    )
    hip_weights = torch.empty_like(weights)
    hip_ids = torch.empty_like(ids)
    topk_ops.biased_grouped_topk_hip(
        logits,
        bias,
        hip_weights,
        hip_ids,
        1,
        1,
        need_renorm,
        scale,
    )

    assert torch.equal(ids, expected_ids)
    assert torch.equal(ids, hip_ids)
    torch.testing.assert_close(
        weights,
        expected_weights,
        atol=2e-5,
        rtol=2e-5,
        equal_nan=True,
    )
    torch.testing.assert_close(
        weights,
        hip_weights,
        atol=2e-5,
        rtol=2e-5,
        equal_nan=True,
    )


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("need_renorm", [True, False])
@pytest.mark.parametrize(
    "experts,topk,rows",
    [
        (512, 5, 16),  # Top-(4+1) HERD candidates
        (768, 9, 8),  # Top-(8+1) HERD candidates
        (896, 16, 64),  # Kimi-K3 routing
        (896, 17, 64),  # Kimi-K3 HERD candidates
        (1024, 17, 128),
    ],
)
def test_exact_sigmoid_topk_herd_widths(experts, topk, rows, need_renorm, bias_dtype):
    """The Triton kernel pads K+1 widths to a power of two and masks the
    stores, with a BF16 or an FP32 bias (Kimi-K3's). biased_grouped_topk sends
    these aligned rows to the HIP multi-wave kernel, so the test calls the
    Triton kernel itself, checks it against the HIP one, and then checks that
    biased_grouped_topk gives the HIP results."""
    torch.manual_seed(experts * topk + rows)
    logits = torch.randn((rows, experts), dtype=torch.bfloat16, device="cuda")
    bias = (torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1).to(
        bias_dtype
    )
    valid = rows - 3
    num_valid_rows = torch.tensor([valid], dtype=torch.int32, device="cuda")
    expected_weights, expected_ids = _torch_reference(
        logits, bias, topk, need_renorm, 2.5
    )
    expected_ids[valid:] = -1
    expected_weights[valid:] = 0.0
    # The row stride covers every padded slot, so a store past topk shows up
    # in the gap instead of in the next row.
    stride = 2 * topk
    weights_buf = torch.full((rows, stride), -7.0, dtype=torch.float32, device="cuda")
    ids_buf = torch.full((rows, stride), -7, dtype=torch.int32, device="cuda")
    weights, ids = weights_buf[:, :topk], ids_buf[:, :topk]

    # biased_grouped_topk sends these rows to the HIP multi-wave kernel.
    assert not topk_ops._can_use_gfx1250_exact_sigmoid_topk(logits, bias, weights, ids)
    topk_ops._gfx1250_exact_sigmoid_topk(
        logits, bias, weights, ids, need_renorm, 2.5, num_valid_rows
    )

    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights, atol=2e-5, rtol=2e-5)
    assert torch.all(ids_buf[:, topk:] == -7)
    assert torch.all(weights_buf[:, topk:] == -7.0)
    hip_weights = torch.empty_like(weights)
    hip_ids = torch.empty_like(ids)
    topk_ops.biased_grouped_topk_hip(
        logits, bias, hip_weights, hip_ids, 1, 1, need_renorm, 2.5, num_valid_rows
    )
    assert torch.equal(ids, hip_ids)
    torch.testing.assert_close(weights, hip_weights, atol=1e-6, rtol=1e-6)

    # The public entry gives the HIP kernel's bits.
    weights.fill_(-7.0)
    ids.fill_(-7)
    topk_ops.biased_grouped_topk(
        logits, bias, weights, ids, 1, 1, need_renorm, 2.5, num_valid_rows
    )
    assert torch.equal(ids, hip_ids)
    assert torch.equal(weights, hip_weights)
    assert torch.all(ids_buf[:, topk:] == -7)
    assert torch.all(weights_buf[:, topk:] == -7.0)


@pytest.mark.parametrize(
    "experts,weights_pitch",
    [
        (512, 9),  # The HIP op writes both outputs with the row stride of ids.
        (900, 8),  # No multi-wave instance; contiguous rows.
    ],
)
def test_exact_sigmoid_topk_triton_route(experts, weights_pitch, monkeypatch):
    """Rows of whole quads that the multi-wave kernel cannot take stay on the
    Triton kernel, with the reference's ids and weights."""
    torch.manual_seed(experts + weights_pitch)
    rows, topk = 32, 8
    logits = torch.randn((rows, experts), dtype=torch.bfloat16, device="cuda")
    bias = (torch.randn(experts, device="cuda") * 0.1).to(torch.bfloat16)
    weights = torch.full((rows, weights_pitch), -7.0, device="cuda")[:, :topk]
    ids = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")

    def no_hip(*args):
        raise AssertionError("biased_grouped_topk took the HIP kernel")

    monkeypatch.setattr(topk_ops, "biased_grouped_topk_hip", no_hip)
    topk_ops.biased_grouped_topk(logits, bias, weights, ids, 1, 1, True, 2.5)

    expected_weights, expected_ids = _torch_reference(logits, bias, topk, True, 2.5)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights, atol=2e-5, rtol=2e-5)


def _unaligned_copy(t: torch.Tensor, pad: int) -> torch.Tensor:
    """t's values in a view one element past an aligned base; a 2-D t gets a
    row pitch of t.shape[1] + pad elements."""
    rows, cols = (1, t.shape[0]) if t.ndim == 1 else t.shape
    pitch = cols + pad
    view = t.new_zeros(rows * pitch + 1)[1:].view(rows, pitch)[:, :cols]
    view.copy_(t.view(rows, cols))
    return view[0] if t.ndim == 1 else view


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", ["random", "ties"])
@pytest.mark.parametrize(
    "experts,topk,rows",
    [(512, 8, 1), (768, 9, 8), (896, 16, 64), (896, 17, 64), (1024, 16, 128)],
)
def test_exact_sigmoid_topk_unaligned_base(experts, topk, rows, case, bias_dtype):
    """Logits and bias one element past an aligned address, with a row pitch of
    whole quads, take the HIP multi-wave kernel. It gives the bits it gives for
    an aligned copy, the torch reference's ids and the Triton kernel's."""
    torch.manual_seed(experts + topk + rows)
    logits = torch.randn((rows, experts), dtype=torch.bfloat16, device="cuda")
    bias = (torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1).to(
        bias_dtype
    )
    if case == "ties":
        logits[:, ::3] = 0.0
        bias[::5] = 0.0
    u_logits, u_bias = _unaligned_copy(logits, 4), _unaligned_copy(bias, 0)
    assert u_logits.stride(0) % 4 == 0
    assert u_logits.data_ptr() % (4 * u_logits.element_size()) != 0
    assert u_bias.data_ptr() % (4 * u_bias.element_size()) != 0

    def outputs():
        return (
            torch.full((rows, topk), -7.0, dtype=torch.float32, device="cuda"),
            torch.full((rows, topk), -7, dtype=torch.int32, device="cuda"),
        )

    weights, ids = outputs()
    assert not topk_ops._can_use_gfx1250_exact_sigmoid_topk(
        u_logits, u_bias, weights, ids
    )
    topk_ops.biased_grouped_topk(u_logits, u_bias, weights, ids, 1, 1, True, 2.5)

    aligned_w, aligned_ids = outputs()
    topk_ops.biased_grouped_topk_hip(
        logits, bias, aligned_w, aligned_ids, 1, 1, True, 2.5
    )
    assert torch.equal(ids, aligned_ids)
    assert torch.equal(weights, aligned_w)

    expected_weights, expected_ids = _torch_reference(logits, bias, topk, True, 2.5)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights, atol=2e-5, rtol=2e-5)

    triton_w, triton_ids = outputs()
    topk_ops._gfx1250_exact_sigmoid_topk(
        u_logits, u_bias, triton_w, triton_ids, True, 2.5, None
    )
    assert torch.equal(ids, triton_ids)
    torch.testing.assert_close(weights, triton_w, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("unaligned", [False, True])
def test_exact_sigmoid_topk_compiled_call(unaligned):
    """biased_grouped_topk is an opaque custom op, so a compiled call runs the
    HIP multi-wave kernel as an eager one does and returns the same bits."""
    torch.manual_seed(0)
    logits = torch.randn((64, 896), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.1
    if unaligned:
        logits, bias = _unaligned_copy(logits, 4), _unaligned_copy(bias, 0)
    eager_w = torch.full((64, 17), -7.0, dtype=torch.float32, device="cuda")
    eager_ids = torch.full((64, 17), -7, dtype=torch.int32, device="cuda")
    assert not topk_ops._can_use_gfx1250_exact_sigmoid_topk(
        logits, bias, eager_w, eager_ids
    )
    topk_ops.biased_grouped_topk(logits, bias, eager_w, eager_ids, 1, 1, True, 2.5)

    weights, ids = torch.empty_like(eager_w), torch.empty_like(eager_ids)
    for dynamic in (False, True):
        weights.fill_(-7.0)
        ids.fill_(-7)
        torch._dynamo.reset()
        torch.compile(topk_ops.biased_grouped_topk, fullgraph=True, dynamic=dynamic)(
            logits, bias, weights, ids, 1, 1, True, 2.5
        )
        assert torch.equal(ids, eager_ids), f"dynamic={dynamic}"
        assert torch.equal(weights, eager_w), f"dynamic={dynamic}"

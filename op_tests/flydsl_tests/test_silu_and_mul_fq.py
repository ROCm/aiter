import pytest
import torch

from aiter import dtypes
from aiter.ops.flydsl.moe_kernels import _get_compiled_silu_fused, _run_compiled
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort


def _swiglu_ref(x: torch.Tensor, inter_dim: int) -> torch.Tensor:
    gate, up = x.float().split([inter_dim, inter_dim], dim=-1)
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)


@pytest.mark.parametrize("token_num, topk, inter_dim", [(4, 2, 256)])
def test_flydsl_swiglu_fused_fp4_quant_matches_reference(
    token_num: int, topk: int, inter_dim: int
):
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("MXFP4 is not available in this torch build")
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    rows = token_num * topk

    gate = torch.linspace(-8.0, 8.0, rows * inter_dim, device=device).reshape(
        rows, inter_dim
    )
    up = torch.linspace(-2.0, 2.0, rows * inter_dim, device=device).reshape(
        rows, inter_dim
    )
    x = torch.cat([gate, up], dim=-1).to(torch.bfloat16)

    token_ids = torch.arange(token_num, device=device).repeat_interleave(topk)
    slot_ids = torch.arange(topk, device=device).repeat(token_num)
    order = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4], device=device)
    sorted_ids = ((slot_ids[order] << 24) | token_ids[order]).to(torch.int32)
    num_valid_ids = torch.tensor([rows], dtype=torch.int32, device=device)

    out = torch.empty(
        (token_num, topk, inter_dim // 2), dtype=dtypes.fp4x2, device=device
    )
    scale_cols = inter_dim // 32
    padded_rows = (sorted_ids.numel() + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted = torch.empty(
        padded_rows * padded_cols, dtype=torch.uint8, device=device
    )

    kernel = _get_compiled_silu_fused(
        inter_dim,
        topk,
        quant_mode="fp4",
        gui_layout=False,
        act="swiglu",
        enable_bias=False,
    )
    _run_compiled(
        kernel,
        (
            x,
            out.view(-1).view(torch.uint8),
            out_scale_sorted,
            sorted_ids,
            num_valid_ids,
            sorted_ids,
            torch.empty(0, dtype=torch.float32, device=device),
            token_num,
            sorted_ids.numel(),
            torch.cuda.current_stream(),
        ),
    )

    ref_act = _swiglu_ref(x, inter_dim)
    ref_out, ref_scale = per_1x32_f4_quant(ref_act, quant_dtype=dtypes.fp4x2)
    ref_scale_sorted = moe_mxfp4_sort(
        ref_scale.view(token_num, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
    )

    ref_out = ref_out.view(token_num, topk, -1)
    torch.testing.assert_close(out.view(torch.uint8), ref_out.view(torch.uint8))
    ref_scale_bytes = ref_scale_sorted.view(torch.uint8).view(-1)
    scale_positions = []
    n32_sort = scale_cols * 32
    for row in range(rows):
        for col in range(scale_cols):
            scale_positions.append(
                (row >> 5) * n32_sort
                + (col >> 3) * 256
                + (col & 3) * 64
                + (row & 15) * 4
                + ((col >> 2) & 1) * 2
                + ((row >> 4) & 1)
            )
    scale_positions = torch.tensor(scale_positions, dtype=torch.long, device=device)
    torch.testing.assert_close(
        out_scale_sorted[scale_positions],
        ref_scale_bytes[scale_positions],
    )

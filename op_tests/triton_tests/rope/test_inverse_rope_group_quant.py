import pytest
import torch

from aiter.ops.triton.rope.inverse_rope_group_quant import inverse_rope_group_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


def make_inputs(S, H, head_dim, rope_dim, max_pos=4096, pos_dtype=torch.int64):
    torch.manual_seed(7)
    o = torch.randn((S, H, head_dim), device="cuda").to(torch.bfloat16)
    positions = torch.randint(0, max_pos, (S,), device="cuda").to(pos_dtype)
    angle = torch.randn((max_pos, rope_dim // 2), device="cuda")
    # cos and sin are views of one [max_pos, rope_dim] table, as a model keeps it.
    cos_sin = torch.cat((angle.cos(), angle.sin()), dim=-1)
    return o, positions, cos_sin[:, : rope_dim // 2], cos_sin[:, rope_dim // 2 :]


def run_torch(o, positions, cos, sin, num_groups, quant_group_size, fp8_dtype):
    """Inverse GPT-J RoPE on the rope tail, then FP8 group quant with E8M0 scales
    2^ceil(log2(amax / fp8_max))."""
    S, H, D = o.shape
    rope_dim = 2 * cos.shape[-1]
    x = o.float().clone()
    c = cos[positions.long()].float()[:, None, :]
    s = sin[positions.long()].float()[:, None, :]
    even = x[..., D - rope_dim :: 2].clone()
    odd = x[..., D - rope_dim + 1 :: 2].clone()
    x[..., D - rope_dim :: 2] = even * c + odd * s
    x[..., D - rope_dim + 1 :: 2] = odd * c - even * s
    groups = x.reshape(S, H * D // quant_group_size, quant_group_size)
    amax = groups.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    fp8_max = torch.finfo(fp8_dtype).max
    bits = (torch.ceil(torch.log2(amax / fp8_max)) + 127).clamp(0, 254)
    q = (groups * torch.exp2(127 - bits)[..., None]).to(fp8_dtype)
    return (
        q.reshape(S, num_groups, H * D // num_groups),
        bits.to(torch.uint8).reshape(S, num_groups, -1),
    )


@pytest.mark.parametrize(
    "S, H, head_dim, rope_dim",
    [(1, 1, 128, 64), (7, 16, 512, 64), (64, 64, 512, 64), (513, 64, 512, 64)],
)
@pytest.mark.parametrize("num_groups", [1, 8])
@pytest.mark.parametrize("quant_group_size", [32, 128])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_inverse_rope_group_quant(
    S, H, head_dim, rope_dim, num_groups, quant_group_size, pos_dtype
):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 is not available on this architecture")
    if (H * head_dim) % (num_groups * quant_group_size):
        pytest.skip("each group must hold whole quant groups")

    o, positions, cos, sin = make_inputs(S, H, head_dim, rope_dim, pos_dtype=pos_dtype)
    x_fp8, x_scale = inverse_rope_group_quant(
        o, positions, cos, sin, num_groups, quant_group_size
    )
    ref_fp8, ref_scale = run_torch(
        o, positions, cos, sin, num_groups, quant_group_size, get_fp8_e4m3_dtype()
    )

    assert x_fp8.dtype == get_fp8_e4m3_dtype() and x_fp8.shape == ref_fp8.shape
    assert x_scale.shape == ref_scale.shape
    # A fused multiply-add can move a rotated value by one fp32 ulp, which very
    # rarely flips its FP8 rounding. Allow a tiny fraction of such bytes.
    data_diff = (x_fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).float().mean()
    scale_diff = (x_scale.view(torch.uint8) != ref_scale).float().mean()
    assert data_diff.item() <= 1e-4, f"{data_diff.item():.2e} of the FP8 bytes differ"
    assert scale_diff.item() <= 1e-4, f"{scale_diff.item():.2e} of the scales differ"


def test_inverse_rope_group_quant_preallocated_out():
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 is not available on this architecture")
    S, H, head_dim, rope_dim, G = 9, 64, 512, 64, 8
    o, positions, cos, sin = make_inputs(S, H, head_dim, rope_dim)
    x_fp8 = torch.empty(
        (S, G, H * head_dim // G), dtype=get_fp8_e4m3_dtype(), device="cuda"
    )
    x_scale = torch.empty(
        (S, G, H * head_dim // G // 32), dtype=torch.float8_e8m0fnu, device="cuda"
    )
    out_fp8, out_scale = inverse_rope_group_quant(
        o, positions, cos, sin, G, 32, x_fp8=x_fp8, x_scale=x_scale
    )
    ref_fp8, ref_scale = inverse_rope_group_quant(o, positions, cos, sin, G, 32)

    assert out_fp8.data_ptr() == x_fp8.data_ptr()
    assert out_scale.data_ptr() == x_scale.data_ptr()
    assert torch.equal(out_fp8.view(torch.uint8), ref_fp8.view(torch.uint8))
    assert torch.equal(out_scale.view(torch.uint8), ref_scale.view(torch.uint8))


def test_inverse_rope_group_quant_rejects_other_layouts():
    o, positions, cos, sin = make_inputs(4, 8, 128, 64)
    with pytest.raises(NotImplementedError):
        inverse_rope_group_quant(o, positions, cos, sin, 1, 32, scale_layout="n32k4")

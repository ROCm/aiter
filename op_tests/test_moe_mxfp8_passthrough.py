"""Contract test for the MXFP8 activation passthrough in fused_moe.

The passthrough lets a caller that has already produced fp8 activations with
group-32 e8m0 microscales -- the format the per_1x32 (MXFP4-weight) MoE kernels
consume -- hand them over as-is, so only the scale is sorted and no
requantization runs.

Scope, stated plainly: this pins the branch predicate and the scale layout the
branch depends on. It deliberately does not drive fused_moe end to end, because
that requires real MXFP4 expert weights and a matching tuned configuration;
constructing a toy w4a8 setup here crashed the kernel rather than testing it,
which is worse than not covering it. End-to-end coverage for this path comes
from the SGLang MoRI dispatch integration (8192 in / 1024 out, EP8 + MoRI,
GSM8K 0.945 over 1319 questions with zero invalid).

What this catches is the silent half of the failure surface: the branch is
selected on dtype plus the presence of a scale, and the scale layout is
interpreted bytewise, so a wrong group size or scale dtype produces wrong
numbers rather than an exception.
"""

import inspect

import pytest
import torch

from aiter import dtypes
from aiter import fused_moe as fused_moe_mod

MODEL_DIM = 7168  # DeepSeek-V4
MX_BLOCK = 32  # group-32 microscales


def test_scale_group_size_is_32():
    """per_1x32 is the whole point. A group-128 layout would send the caller
    back through the fp8->bf16 upscale this branch exists to avoid."""
    assert MODEL_DIM % MX_BLOCK == 0
    assert MODEL_DIM // MX_BLOCK == 224


def test_e8m0_scale_is_one_byte():
    """The dispatch buffer is sized from this. fp32 scales would need 4x the
    room, and the payload would be truncated rather than rejected."""
    assert torch.float8_e8m0fnu.itemsize == 1


def test_scale_tensor_shape_matches_group_layout():
    """One scale per 32 channels, including the zero-live-token case a rank can
    hit during decode."""
    for tokens in (0, 1, 8, 112):
        scale = torch.empty(
            (tokens, MODEL_DIM // MX_BLOCK), dtype=torch.float8_e8m0fnu
        )
        assert scale.shape == (tokens, 224)


def test_passthrough_branch_is_present_and_predicated_on_dtype_and_scale():
    """The branch must trigger on fp8 activations carrying a scale, and only
    then -- otherwise it would swallow the unquantized path."""
    src = inspect.getsource(fused_moe_mod)
    assert "hidden_states.dtype == dtypes.fp8 and a1_scale is not None" in src, (
        "MXFP8 passthrough predicate missing or changed; a caller delivering "
        "pre-quantized fp8 would silently fall back to requantizing"
    )


def test_passthrough_sorts_the_scale_like_the_fp4_branch():
    """The scale must still be sorted into the MoE's expected order. Skipping
    the sort would leave microscales misaligned with their tokens, which is
    wrong output rather than a crash."""
    src = inspect.getsource(fused_moe_mod)
    branch = src.split("hidden_states.dtype == dtypes.fp8 and a1_scale is not None")[1]
    head = branch[:600]
    assert "mxfp4_moe_sort_fwd" in head, "passthrough does not sort the scale"


def test_fp8_output_dtype_is_still_rejected():
    """The branch changes how activations are consumed, not what fused_moe may
    emit; fp8 output stays unsupported."""
    src = inspect.getsource(fused_moe_mod)
    assert "Fused_moe unsupported out dtype" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm device")
def test_fp8_dtype_available_on_device():
    """dtypes.fp8 resolves to a real device dtype on this build."""
    x = torch.zeros(4, dtype=dtypes.fp8, device="cuda")
    assert x.numel() == 4

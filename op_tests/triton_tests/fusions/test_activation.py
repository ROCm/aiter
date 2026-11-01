import pytest
import torch
import torch.nn.functional as F

from aiter import logger
from aiter.ops.quant import per_1x32_f4_quant_hip
from aiter.ops.triton.fusions.activation import act_mul_and_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, unshuffle_scale_gemm
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
from op_tests.triton_tests.utils.activation_ref import silu_exp2_ref


def pad_tensor_2d(tensor, mult_m=256, mult_n=8):
    M, N = tensor.shape

    pad_rows = (mult_m - (M % mult_m)) % mult_m
    pad_cols = (mult_n - (N % mult_n)) % mult_n
    padded_tensor = torch.nn.functional.pad(
        tensor, (0, pad_cols, 0, pad_rows), mode="constant", value=0
    )

    return padded_tensor


def torch_act_mul_and_mxfp4_quant(
    input: torch.Tensor, activation: str, shuffle: bool
) -> torch.Tensor:
    """
    The fused kernel casts the original input to float32 and does all the arithmetic
    and bit operations in float32.
    """
    input = input.to(torch.float32)
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    if activation == "silu":
        out = F.silu(x) * y
    elif activation == "gelu":
        out = F.gelu(x) * y
    else:
        out = F.gelu(x, approximate="tanh") * y
    out, out_scale = torch_dynamic_mxfp4_quant(out)
    if shuffle:
        # out_scale_pad = out_scale
        M = out_scale.shape[0]
        N = out.shape[1] * 2
        scaleM = (M + 255) // 256 * 256
        scaleN_valid = (N + 31) // 32
        scaleN = (scaleN_valid + 7) // 8 * 8
        out_scale_pad = torch.empty(
            (scaleM, scaleN), dtype=out_scale.dtype, device=out_scale.device
        )
        out_scale_pad[:M, :scaleN] = out_scale[:M, :scaleN]
        out_scale = shuffle_scale_gemm(
            out_scale_pad, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
        out_scale = out_scale.view(out_scale.shape[0] * 32, -1)
    return out, out_scale


@pytest.mark.parametrize(
    "M, N",
    [
        (512, 57344),
        (504, 57344),
        (1, 57344),
        (4, 57344),
        (32, 8192),
        (1, 4),
        (1, 28),
        (1, 32),
        (1, 64),
        (1, 68),
        (2, 4),
        (2, 28),
        (2, 32),
        (2, 64),
        (2, 68),
        (128, 4),
        (128, 28),
        (128, 32),
        (128, 64),
        (128, 68),
        (256, 32),
        (256, 512),
        (256, 1024),
        (160, 40),
        (280, 20),
        (32, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("scale_shuffle_padding", [False, True])
def test_act_mul_and_mxfp4_quant(
    M: int, N: int, dtype, activation: str, shuffle: bool, scale_shuffle_padding: bool
):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    if shuffle and N % 512 != 0:
        pytest.skip()

    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    logger.debug("x.shape=%s x=%s", x.shape, x)

    triton_out, triton_scale = act_mul_and_mxfp4_quant(
        x,
        activation=activation,
        shuffle=shuffle,
        scale_shuffle_padding=scale_shuffle_padding,
    )
    logger.debug("triton_out.shape=%s triton_out=%s", triton_out.shape, triton_out)
    logger.debug(
        "triton_scale.shape=%s triton_scale=%s", triton_scale.shape, triton_scale
    )

    torch_out, torch_scale = torch_act_mul_and_mxfp4_quant(
        x, activation=activation, shuffle=shuffle
    )

    if shuffle:
        triton_scale = unshuffle_scale_gemm(
            triton_scale.view(triton_scale.shape[0] // 32, -1), arch="gfx950"
        )
        torch_scale = unshuffle_scale_gemm(
            torch_scale.view(torch_scale.shape[0] // 32, -1), arch="gfx950"
        )

    logger.debug("torch_out.shape=%s torch_out=%s", torch_out.shape, torch_out)
    logger.debug("torch_scale.shape=%s torch_scale=%s", torch_scale.shape, torch_scale)

    scaleN_valid = (N // 2 + 31) // 32
    triton_scale = triton_scale[:M, :scaleN_valid]
    torch_scale = torch_scale[:M, :scaleN_valid]

    torch.testing.assert_close(triton_out, torch_out)
    torch.testing.assert_close(triton_scale, torch_scale)


def torch_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up as an unfused kernel that writes x.dtype computes it:
    silu in fp32 (the Triton kernels' exp2 form) rounded to x.dtype, then the
    product in fp32 rounded again."""
    gate, up = x.chunk(2, dim=-1)
    return (silu_exp2_ref(gate).to(x.dtype).float() * up.float()).to(x.dtype)


@pytest.mark.parametrize("M", [1, 2, 8, 9, 32, 33, 255, 256, 257, 2048])
@pytest.mark.parametrize("N", [256, 8704, 17408])
@pytest.mark.parametrize("magnitude", [1.0, 64.0])
@pytest.mark.parametrize("shuffle", [False, True])
def test_act_mul_and_mxfp4_quant_roundup_matches_hip_quant(
    M: int, N: int, magnitude: float, shuffle: bool
):
    """scaling_mode="roundup" with round_to_input_dtype gives the bytes of a bf16
    silu-and-mul followed by per_1x32_f4_quant_hip: the fp4 values of the M rows
    and the scales they own. N is the output width."""
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.manual_seed(M)
    x = torch.randn((M, 2 * N), dtype=torch.float32, device="cuda")
    x = (x * magnitude).to(torch.bfloat16)
    # An all-zero and a tiny output block, both under the HIP quant's amax floor.
    x[0, :32] = 0
    x[-1, N - 32 : N] = 1e-12
    x[-1, -32:] = 1e-12

    out, scale = act_mul_and_mxfp4_quant(
        x, "silu", scaling_mode="roundup", shuffle=shuffle, round_to_input_dtype=True
    )
    ref_out, ref_scale = per_1x32_f4_quant_hip(torch_silu_and_mul(x), shuffle=shuffle)
    ref_scale = ref_scale.view(torch.uint8)

    if shuffle:
        # The padding of the shuffled scales is not compared, only the M rows'.
        scale = unshuffle_scale_gemm(
            scale.view(scale.shape[0] // 32, -1), arch="gfx950"
        )
        ref_scale = unshuffle_scale_gemm(
            ref_scale.view(ref_scale.shape[0] // 32, -1), arch="gfx950"
        )
    torch.testing.assert_close(out, ref_out.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(
        scale[:M, : N // 32], ref_scale[:M, : N // 32], atol=0, rtol=0
    )


def test_act_mul_and_mxfp4_quant_scaling_modes():
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    # Two output blocks of silu(20) * up: 6.484375, whose scale "even" rounds
    # down to 2^0 and "ceil" up to 2^1, and 20 * 2^-70, below the 1e-10 amax
    # floor of "roundup" (scale 2^-35).
    x = torch.full((1, 128), 20.0, dtype=torch.bfloat16, device="cuda")
    x[0, 64:96] = 0.32421875
    x[0, 96:] = 2**-70
    for scaling_mode, scale_bytes in (
        ("even", [127, 59]),
        ("ceil", [128, 59]),
        ("roundup", [128, 92]),
    ):
        _, scale = act_mul_and_mxfp4_quant(x, "silu", scaling_mode=scaling_mode)
        assert scale[0].tolist() == scale_bytes, scaling_mode


def test_act_mul_and_mxfp4_quant_round_to_input_dtype():
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    # silu(20) * 0.30078125 = 6.015625 in fp32, a tie that bf16 rounds to 6.0:
    # round_to_input_dtype moves the "roundup" scale from 2^1 to 2^0.
    x = torch.full((1, 64), 20.0, dtype=torch.bfloat16, device="cuda")
    x[0, 32:] = 0.30078125
    for round_to_input_dtype, scale_byte in ((False, 128), (True, 127)):
        _, scale = act_mul_and_mxfp4_quant(
            x,
            "silu",
            scaling_mode="roundup",
            round_to_input_dtype=round_to_input_dtype,
        )
        assert scale[0, 0].item() == scale_byte, round_to_input_dtype


def test_act_mul_and_mxfp4_quant_rejects_unknown_scaling_mode():
    x = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="scaling_mode"):
        act_mul_and_mxfp4_quant(x, "silu", scaling_mode="round_up")

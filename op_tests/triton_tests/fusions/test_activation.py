import pytest
import torch
import torch.nn.functional as F

from aiter import logger
from aiter.ops.triton.fusions import activation as activation_ops
from aiter.ops.triton.fusions.activation import (
    act_mul_and_fp8_group_quant,
    act_mul_and_mxfp4_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, unshuffle_scale_gemm
from op_tests.triton_tests.quant.test_fused_fp8_quant import (
    run_torch_reduce_act_mul_fp8_group_quant,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dynamic_mxfp4_quant


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


# Only 17408 fills whole 1024-column tiles; 1000 also ends in a partial group.
FP8_GROUP_QUANT_N = [2 * 1000, 2 * 2176, 2 * 4352, 2 * 8704, 2 * 17408]


@pytest.mark.parametrize("M", [1, 32, 256, 4097])
@pytest.mark.parametrize("N", FP8_GROUP_QUANT_N)
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_act_mul_and_fp8_group_quant(M: int, N: int, activation: str):
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    out_q, out_s = act_mul_and_fp8_group_quant(x, activation, group_size=128)
    # The kernel computes in fp32, so the reference also does.
    (ref_q, ref_s), _ = run_torch_reduce_act_mul_fp8_group_quant(
        x.float(), None, activation, torch.float32, out_q.dtype, group_size=128
    )

    torch.testing.assert_close(out_s, ref_s, atol=0, rtol=1e-5)
    # A value at a rounding boundary can move by one FP8 step.
    torch.testing.assert_close(out_q.float(), ref_q.float(), atol=2**-9, rtol=0.125)


class _LaunchSpy:
    def __init__(self, kernel):
        self.kernel = kernel
        self.kwargs = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.kwargs = kwargs
            return self.kernel[grid](*args, **kwargs)

        return launch


@pytest.mark.parametrize("M", [1, 4097])
@pytest.mark.parametrize("N", FP8_GROUP_QUANT_N)
def test_act_mul_and_fp8_group_quant_launch(monkeypatch, M: int, N: int):
    # gfx942 has a tuned config. Other architectures keep one group for each program.
    config_dir = resolve_config_dir("fusions", "ACT_MUL_AND_FP8_GROUP_QUANT")
    configs = load_config_json(
        f"{config_dir}/DEFAULT.json", required=arch_info.get_arch() == "gfx942"
    )
    spy = _LaunchSpy(activation_ops._act_mul_and_dynamic_fp8_group_quant_kernel)
    monkeypatch.setattr(
        activation_ops, "_act_mul_and_dynamic_fp8_group_quant_kernel", spy
    )
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    act_mul_and_fp8_group_quant(x, "silu", group_size=128)

    if configs is None:
        assert spy.kwargs["BLOCK_SIZE_N"] == 128
        assert "num_warps" not in spy.kwargs
        return
    for key, value in select_leq_config(configs, M, prefix="M_LEQ_").items():
        assert spy.kwargs[key] == value, key

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_flatten_mxfp4_quant,
    fused_rms_mxfp4_quant,
    fused_reduce_act_mul_and_mxfp4_quant,
    fused_reduce_rms_mxfp4_quant,
    fused_dynamic_mxfp4_quant_moe_sort,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    mxfp4_to_f32,
    e8m0_to_f32,
    SCALE_GROUP_SIZE,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    shuffle_scales,
    un_shuffle_scales,
)
from aiter.ops.quant import per_1x32_f4_quant_hip
from aiter.utility.fp4_utils import moe_mxfp4_sort, dynamic_mxfp4_quant

torch.manual_seed(0)


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps).reshape(-1, 1)
    rms_norm = input * norm_factor * weight.reshape(1, -1)
    return rms_norm


def calculate_target_w_torch(
    x1,
    rms1_w,
    resid1,
    x2,
    rms2_w,
    x3=None,
    eps=1e-6,
    shuffle=False,
    dtype=torch.bfloat16,
):
    out_dtype = dtype if dtype is not None else x1.dtype

    out3 = None
    if x3 is not None:
        x1 = x1.to(torch.float32).sum(axis=0)
        x2 = x2.to(torch.float32).sum(axis=0)
        out3 = x3.to(torch.float32).sum(axis=0).to(out_dtype)

    x1 = x1.to(torch.float32)
    rms1_w = rms1_w.to(torch.float32)
    res1_out = None
    if resid1 is not None:
        resid1 = resid1.to(torch.float32)
        x1 = res1_out = x1 + resid1
        res1_out = res1_out.to(out_dtype)
    x1 = rmsnorm(x1, rms1_w, eps)
    out1 = x1.to(out_dtype)
    out1_fp4, out1_scale = torch_dynamic_mxfp4_quant(x1)

    out2 = None
    if x2 is not None:
        x2 = x2.to(torch.float32)
        rms2_w = rms2_w.to(torch.float32)
        out2 = rmsnorm(x2, rms2_w, eps).to(out_dtype)

    if shuffle:
        out1_scale_pad = out1_scale
        M = out1_scale.shape[0]
        N = x1.shape[1]
        scaleM = (M + 255) // 256 * 256
        scaleN_valid = (N + 31) // 32
        scaleN = (scaleN_valid + 7) // 8 * 8
        out1_scale_pad = torch.empty(
            (scaleM, scaleN), dtype=out1_scale.dtype, device=out1_scale.device
        )
        out1_scale_pad[:M, :scaleN_valid] = out1_scale[:M, :scaleN_valid]
        out1_scale = shuffle_scales(out1_scale_pad)
        out1_scale = out1_scale.view(out1_scale.shape[0] * 32, -1)

    if x3 is not None:
        return (out1_fp4, out1_scale), out1, out2, res1_out, out3
    return (out1_fp4, out1_scale), out1, out2, res1_out


def convert_mxfp4_to_fp32(x, x_scales):
    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)[:, : x_f32.shape[1]]
    x_f32 = x_f32 * x_scales_f32
    return x_f32


def generate_fused_rms_quant_data(
    x1_shape=(32, 1536),
    x1_stride=(2112, 1),
    x2_shape=(32, 512),
    x2_stride=(2112, 1),
    inp2=False,
    res1=False,
    dtype=torch.bfloat16,
):
    x1 = torch.randn((x1_shape[0], x1_stride[0]), dtype=dtype, device="cuda")
    x1 = x1[:, : x1_shape[1]]
    x2 = None
    rms2_w = None
    if inp2:
        x2 = torch.randn((x2_shape[0], x2_stride[0]), dtype=dtype, device="cuda")
        x2 = x2[:, : x2_shape[1]]
        rms2_w = torch.randn(x2.shape[1], dtype=dtype, device="cuda")

    rms1_w = torch.randn(x1.shape[1], dtype=dtype, device="cuda")
    resid1 = None
    if res1:
        resid1 = torch.randn_like(x1, dtype=dtype, device="cuda")
    return x1, x2, rms1_w, rms2_w, resid1


@pytest.mark.parametrize("B", [1, 4, 16, 32, 1000, 10000])
@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flatten_quant(B: int, M: int, N: int, dtype):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x = torch.randn((B, M, N), dtype=dtype, device="cuda").transpose(0, 1)

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x.flatten(1, 2))
    triton_out, triton_scale = fused_flatten_mxfp4_quant(x)

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)


@pytest.mark.parametrize(
    "M, N1, N2, stride",
    [
        (M, N1, N2, stride)
        for M in [1, 4, 33, 64, 132, 256]  # TODO: debug for 131072
        for N1, N2, stride in [
            (200, 200, 200),
            (256, 256, 256),
            (256, 256, 2112),
        ]
    ],
)
@pytest.mark.parametrize("inp2", [True, False])
@pytest.mark.parametrize("res1", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("scale_shuffle_padding", [True, False])
def test_fused_rms_quant(
    M: int,
    N1: int,
    N2: int,
    stride: int,
    inp2: bool,
    res1: bool,
    dtype,
    shuffle: bool,
    scale_shuffle_padding: bool,
):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    x1, x2, rms1_w, rms2_w, resid1 = generate_fused_rms_quant_data(
        x1_shape=(M, N1),
        x2_shape=(M, N2),
        x1_stride=(stride, 1),
        x2_stride=(stride, 1),
        inp2=inp2,
        res1=res1,
        dtype=dtype,
    )
    (y1_fp4_torch, y1_scales_torch), y1_torch, y2_torch, y1_res_torch = (
        calculate_target_w_torch(
            x1, rms1_w, resid1, x2, rms2_w, shuffle=shuffle, dtype=dtype
        )
    )

    (y1_fp4_triton, y1_scales_triton), y1_triton, y2_triton, y1_res_triton = (
        fused_rms_mxfp4_quant(
            x1,
            rms1_w,
            1e-6,
            x2,
            rms2_w,
            1e-6,
            resid1,
            shuffle=shuffle,
            scale_shuffle_padding=scale_shuffle_padding,
            output_unquantized_inp1=True,
        )
    )

    if y1_triton is not None:
        torch.testing.assert_close(y1_torch, y1_triton)

    if shuffle:
        y1_scales_triton = un_shuffle_scales(
            y1_scales_triton.view(y1_scales_triton.shape[0] // 32, -1)
        )
        y1_scales_torch = un_shuffle_scales(
            y1_scales_torch.view(y1_scales_torch.shape[0] // 32, -1)
        )

    scaleN_valid = (N1 + 31) // 32
    y1_scales_triton = y1_scales_triton[:M, :scaleN_valid]
    y1_scales_torch = y1_scales_torch[:M, :scaleN_valid]

    if y2_triton is not None:
        torch.testing.assert_close(y2_torch, y2_triton)

    if y1_res_triton is not None:
        torch.testing.assert_close(y1_res_torch, y1_res_triton)

    y1_fp32_torch = convert_mxfp4_to_fp32(y1_fp4_torch, y1_scales_torch)
    y1_fp32_triton = convert_mxfp4_to_fp32(y1_fp4_triton, y1_scales_triton)

    torch.testing.assert_close(y1_fp32_torch, y1_fp32_triton)


def run_torch_reduce_act_mul_mxfp4_group_quant(x, x2, activation, dtype, shuffle):
    x = x.to(torch.float32)
    d = x.shape[-1] // 2
    y2 = None
    if x.dim() == 3:
        x = x.sum(axis=0)
        y2 = x2.sum(axis=0).to(dtype=dtype)
    else:
        assert x2 is None, "x2 must be None in x.dim() == 2 cases"
    x, x_mul = x.split([d, d], dim=-1)
    if activation == "silu":
        out = F.silu(x) * x_mul
    elif activation == "gelu":
        out = F.gelu(x) * x_mul
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
        out_scale = shuffle_scales(out_scale_pad)
        out_scale = out_scale.view(out_scale.shape[0] * 32, -1)
    return (out, out_scale), y2


def generate_fused_reduce_act_mul_mxfp4_group_quant(
    M: int,
    N1: int,
    dtype=torch.bfloat16,
    SPK: int = 1,
    N2: int = 1,
):
    if SPK == 1:
        x = torch.randn((M, N1 * 2), dtype=dtype).cuda() / 10
    else:
        x = torch.randn((SPK, M, N1 * 2), dtype=torch.float32).cuda() / 10
    x2 = None
    if SPK > 1:
        x2 = torch.randn((SPK, M, N2), dtype=torch.float32).cuda() / 10

    return x, x2


@pytest.mark.parametrize(
    "M, N1, N2",
    [
        (1, 256, 256),
        (2, 256, 256),
        (4, 256, 256),
        (32, 256, 256),
        (1, 4, 256),
        (1, 28, 256),
        (1, 32, 256),
        (1, 64, 256),
        (1, 68, 256),
        (128, 28, 256),
        (128, 32, 256),
        (128, 64, 256),
        (128, 68, 256),
        (256, 32, 256),
    ],
)
@pytest.mark.parametrize("SPK", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("scale_shuffle_padding", [False, True])
def test_fused_reduce_act_mul_mxfp4_group_quant(
    M: int,
    N1: int,
    N2: int,
    SPK: int,
    dtype,
    activation: str,
    shuffle: bool,
    scale_shuffle_padding: bool,
):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    if shuffle and (N1 * 2) % 512 != 0:
        pytest.skip()

    x, x2 = generate_fused_reduce_act_mul_mxfp4_group_quant(
        M, N1, dtype=dtype, SPK=SPK, N2=N2
    )

    (y_q_torch, y_s_torch), y2_torch = run_torch_reduce_act_mul_mxfp4_group_quant(
        x, x2, activation, dtype=dtype, shuffle=shuffle
    )

    (y_q_triton, y_s_triton), y2_triton = fused_reduce_act_mul_and_mxfp4_quant(
        x,
        activation=activation,
        x2=x2,
        shuffle=shuffle,
        scale_shuffle_padding=scale_shuffle_padding,
        dtype=dtype,
    )

    if shuffle:
        y_s_triton = un_shuffle_scales(y_s_triton.view(y_s_triton.shape[0] // 32, -1))
        y_s_torch = un_shuffle_scales(y_s_torch.view(y_s_torch.shape[0] // 32, -1))

    torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)

    scaleN_valid = (N1 // 2 + 31) // 32
    y_s_triton = y_s_triton[:M, :scaleN_valid]
    y_s_torch = y_s_torch[:M, :scaleN_valid]

    torch.testing.assert_close(y_q_triton, y_q_torch)
    torch.testing.assert_close(y_s_triton, y_s_torch)


def generate_fused_reduce_rms_quant_data(M, N1, N2, N3, SPK, dtype=torch.bfloat16):
    if SPK > 1:
        x1 = (
            torch.randn((SPK, M, N1 + N2 + N3), dtype=torch.float32, device="cuda")[
                ..., :N1
            ]
            / 20
        )
        x2 = (
            torch.randn((SPK, M, N1 + N2 + N3), dtype=torch.float32, device="cuda")[
                ..., :N2
            ]
            / 20
        )
        x3 = (
            torch.randn((SPK, M, N1 + N2 + N3), dtype=torch.float32, device="cuda")[
                ..., :N3
            ]
            / 20
        )
    else:
        x1 = torch.randn((M, N1 + N2), dtype=dtype, device="cuda")[..., :N1] / 20
        x2 = torch.randn((M, N1 + N2), dtype=dtype, device="cuda")[..., :N2] / 20
        x3 = None

    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 20
    return x1, w1, x2, w2, res1, x3


@pytest.mark.parametrize("M", [1, 32, 256, 8192])
@pytest.mark.parametrize("N1, N2, N3", [(256, 256, 256), (1536, 512, 64)])
@pytest.mark.parametrize("SPK", [1, 4, 14])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("scale_shuffle_padding", [True, False])
def test_fuse_reduce_rms_quant(
    M: int,
    N1: int,
    N2: int,
    N3: int,
    SPK: int,
    dtype,
    shuffle: bool,
    scale_shuffle_padding: bool,
):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    x1, w1, x2, w2, res1, x3 = generate_fused_reduce_rms_quant_data(
        M, N1, N2, N3, SPK, dtype
    )
    if x3 is None:
        y3_torch = None
        (y1_fp4_torch, y1_scales_torch), y1_torch, y2_torch, y1_res_torch = (
            calculate_target_w_torch(
                x1, w1, res1, x2, w2, x3=x3, shuffle=shuffle, dtype=dtype
            )
        )
    else:
        (y1_fp4_torch, y1_scales_torch), y1_torch, y2_torch, y1_res_torch, y3_torch = (
            calculate_target_w_torch(
                x1, w1, res1, x2, w2, x3=x3, shuffle=shuffle, dtype=dtype
            )
        )

    (
        (y1_fp4_triton, y1_scales_triton),
        y1_triton,
        y2_triton,
        y1_res_triton,
        y3_triton,
    ) = fused_reduce_rms_mxfp4_quant(
        x1,
        w1,
        1e-6,
        x2,
        w2,
        1e-6,
        x3,
        res1,
        shuffle=shuffle,
        scale_shuffle_padding=scale_shuffle_padding,
        output_unquantized_inp1=True,
        dtype=dtype,
    )

    if y1_triton is not None:
        torch.testing.assert_close(y1_torch, y1_triton)

    if y2_triton is not None:
        torch.testing.assert_close(y2_torch, y2_triton)

    if y3_triton is not None:
        torch.testing.assert_close(y3_torch, y3_triton)

    if shuffle:
        y1_scales_triton = un_shuffle_scales(
            y1_scales_triton.view(y1_scales_triton.shape[0] // 32, -1)
        )
        y1_scales_torch = un_shuffle_scales(
            y1_scales_torch.view(y1_scales_torch.shape[0] // 32, -1)
        )

    scaleN_valid = (N1 + 31) // 32
    y1_scales_triton = y1_scales_triton[:M, :scaleN_valid]
    y1_scales_torch = y1_scales_torch[:M, :scaleN_valid]

    if y1_res_triton is not None:
        torch.testing.assert_close(y1_res_torch, y1_res_triton)

    y1_fp32_torch = convert_mxfp4_to_fp32(y1_fp4_torch, y1_scales_torch)
    y1_fp32_triton = convert_mxfp4_to_fp32(y1_fp4_triton, y1_scales_triton)

    tol_fraction = 0.1
    atol = 0.05
    rtol = 0.05
    mismatch_fraction = (
        torch.logical_or(
            torch.abs(y1_fp32_torch - y1_fp32_triton) / y1_fp32_triton > rtol,
            torch.abs(y1_fp32_torch - y1_fp32_triton) > atol,
        )
    ).nonzero().numel() / y1_fp32_triton.numel()
    assert (
        mismatch_fraction < tol_fraction
    ), f"{tol_fraction*100} % of mismatched elements are allowed, there are {mismatch_fraction*100} % of elements mistatched"
    # torch.testing.assert_close(y1_fp32_torch, y1_fp32_triton)


def run_fused_dynamic_mxfp4_quant_moe_sort_ref(
    x,
    sorted_ids,
    token_num,
    topk,
    q_dtype_a,
    num_local_tokens,
    num_valid_ids,
    block_size_M,
):
    x_fp4, x_scales_not_sorted = per_1x32_f4_quant_hip(
        x,
        scale=None,
        quant_dtype=q_dtype_a,
        num_rows=num_local_tokens,
        num_rows_factor=topk,
    )
    x_scales = moe_mxfp4_sort(
        x_scales_not_sorted[: token_num * topk, :].view(token_num, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        block_size=block_size_M,
    )
    return x_fp4, x_scales, x_scales_not_sorted


def run_fused_dynamic_mxfp4_quant_moe_sort_triton(
    x,
    sorted_ids,
    token_num,
    topk,
    q_dtype_a,
    num_local_tokens,
    num_valid_ids,
    block_size_M,
):
    x_fp4, x_scales = fused_dynamic_mxfp4_quant_moe_sort(
        x,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size=block_size_M,
    )
    return x_fp4, x_scales


@pytest.mark.parametrize("hidden_dim", [256])
@pytest.mark.parametrize("token_num", [1, 32, 1024])
@pytest.mark.parametrize(
    "token_num_sort, num_valid_ids_0", [(1, 1), (32, 32), (1024, 1024), (1024, 512)]
)
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_dynamic_mxfp4_quant_moe_sort(
    hidden_dim: int,
    token_num: int,
    token_num_sort: int,
    num_valid_ids_0: int,
    topk: int,
    dtype,
):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")
    q_dtype_a = torch.float4_e2m1fn_x2
    num_local_tokens = None
    num_valid_ids = torch.zeros(2, dtype=torch.int64, device="cuda")
    num_valid_ids[0] = num_valid_ids_0
    num_valid_ids[1] = token_num
    block_size_M = 128

    topk_ids = torch.randint(0, topk, (token_num_sort,), device="cuda")
    topk_ids, _ = torch.sort(topk_ids)
    sorted_ids = torch.randint(0, token_num, (token_num_sort,), device="cuda")
    sorted_ids = (topk_ids << 24) | sorted_ids

    x = torch.randn((token_num, topk, hidden_dim), dtype=dtype, device="cuda") / 20
    x = x.view(-1, hidden_dim)

    x_fp4_ref, x_scales_ref, x_scales_ref_not_sorted = (
        run_fused_dynamic_mxfp4_quant_moe_sort_ref(
            x,
            sorted_ids,
            token_num,
            topk,
            q_dtype_a,
            num_local_tokens,
            num_valid_ids,
            block_size_M,
        )
    )

    x_fp4_triton, x_scales_triton = run_fused_dynamic_mxfp4_quant_moe_sort_triton(
        x,
        sorted_ids,
        token_num,
        topk,
        q_dtype_a,
        num_local_tokens,
        num_valid_ids,
        block_size_M,
    )

    tol = 0.1
    x_scales_ref = x_scales_ref[: num_valid_ids[0]]
    x_scales_triton = x_scales_triton[: num_valid_ids[0]]
    torch.testing.assert_close(
        x_scales_ref.view(torch.uint8),
        x_scales_triton.view(torch.uint8),
        atol=tol,
        rtol=tol,
    )
    # torch.testing.assert_close(x_fp4_ref.view(torch.uint8), x_fp4_triton.view(torch.uint8), atol=tol, rtol=tol)

    _, x_scales_ref_triton_not_sorted = dynamic_mxfp4_quant(x)
    x_scales_ref_triton_not_sorted = x_scales_ref_triton_not_sorted[
        : x_scales_ref_not_sorted.shape[0], : x_scales_ref_not_sorted.shape[1]
    ]
    torch.testing.assert_close(
        x_scales_ref_not_sorted.view(torch.uint8),
        x_scales_ref_triton_not_sorted.view(torch.uint8),
        atol=tol,
        rtol=tol,
    )

    x_ref = convert_mxfp4_to_fp32(
        x_fp4_ref.view(torch.uint8), x_scales_ref_not_sorted.view(torch.uint8)
    )
    x_triton = convert_mxfp4_to_fp32(
        x_fp4_triton.view(torch.uint8), x_scales_ref_triton_not_sorted.view(torch.uint8)
    )
    torch.testing.assert_close(x_ref, x_triton, atol=tol, rtol=tol)

# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_matmul.py

from dataclasses import dataclass, fields
import itertools
import pytest
import torch
from typing import Union
import triton

# routing utilities
from aiter.ops.triton.moe_routing.routing import routing

# matmul utilities
from aiter.ops.triton.moe_op_gemm_a8w4 import (
    moe_gemm_a8w4,
    moe_gemm_torch,
    swizzle_scales,
)

# numerics utilities
from aiter.ops.triton.quant_moe import (
    downcast_to_static_fp8,
    downcast_to_mxfp,
    upcast_from_mxfp,
)

# target-specific utilities
from aiter.ops.triton.utils._triton.arch_info import get_arch

# ---------------
# initialize data
# ---------------


def alloc_rand(shape, device, dtype):
    if dtype.itemsize == 1:
        tmp = 2 ** -(torch.randint(4, 8, shape, device=device, dtype=torch.bfloat16))
        return tmp
    return torch.randn(shape, device=device, dtype=dtype)


def alloc_rand_like(x):
    return alloc_rand(x.shape, x.device, x.dtype)


def init_routing_data(
    m, n_expts_tot, n_expts_act, do_gather, do_scatter, device="cuda"
):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    gather_idx = gather_idx if do_gather else None
    scatter_idx = scatter_idx if do_scatter else None
    # TODO: re-enable
    # if do_gather and do_scatter and n_expts_act == 1 and n_expt_shards == 1:
    #     scatter_idx = mask_indx(scatter_idx, n_expts_act)
    return m, routing_data, gather_idx, scatter_idx


def init_compute_data(
    m,
    n,
    k,
    gindx,
    sindx,
    n_expts_tot,
    n_expts_act,
    act_dtype,
    weight_dtype,
    has_y_gammas,
    device="cuda",
):
    torch.manual_seed(0)
    in_m = m * (n_expts_act if gindx is None else 1)
    shape_x = (in_m, k)
    x = alloc_rand(shape_x, device=device, dtype=act_dtype)
    w = alloc_rand((n_expts_tot, k, n), device=device, dtype=weight_dtype)
    bias = alloc_rand((n_expts_tot, n), device=device, dtype=torch.float32)
    if has_y_gammas:
        gamma = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gamma = None
    return x, w, bias, gamma


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return torch.uint8 if dtype_str == "float4_e2m1" else getattr(torch, dtype_str)


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if ref.numel() == 0:
        return

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert (
        ref.shape == tri.shape
    ), f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(
        inf_mask_ref, inf_mask_tri
    ), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print(
            "%s maximum relative error = %s (threshold = %s)"
            % (description, max_err, maxtol)
        )
        print(
            "%s RMS relative error = %s (threshold = %s)"
            % (description, rms_err, rmstol)
        )

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print(
            "%d / %d mismatched elements (shape = %s) at coords %s"
            % (num_nonzero, rel_err.numel(), tuple(rel_err.shape), bad_idxs.tolist())
        )

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[tuple(bad_idxs)].cpu())
        print("tri values: ", tri[tuple(bad_idxs)].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


# ---------------
# unit tests
# ---------------


@dataclass
class Case:
    m: int
    n: int
    k: int
    act_dtype_str: str
    n_expts_tot: int = 1
    n_expts_act: int = 1
    hbm_swizzling: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            Case(32, 6144, 3072, "float8_e4m3fn", 128, 4, hbm_swizzling=True),
            Case(8192, 3072, 3072, "float8_e4m3fn", 128, 4, hbm_swizzling=True),
            Case(4, 1024, 3072, "float8_e4m3fn", 128, 4, hbm_swizzling=True),
            Case(1024, 3072, 512, "float8_e4m3fn", 128, 4, hbm_swizzling=True),
            Case(4096, 3072, 3072, "float8_e4m3fn", 128, 4),
            Case(16, 1024, 1024, "mxfloat8_e4m3fn", 128, 4, hbm_swizzling=True),
            Case(4096, 1024, 1024, "mxfloat8_e4m3fn", 128, 4),
            Case(16, 256, 256, "mxfloat8_e4m3fn", 128, 4, hbm_swizzling=True),
            Case(4096, 256, 256, "mxfloat8_e4m3fn", 128, 4),
            Case(1000, 704, 800, "mxfloat8_e4m3fn", 8, 2),
            Case(300, 400, 800, "mxfloat8_e4m3fn", 8, 4),
        ]
    ],
)
@pytest.mark.parametrize(
    "do_gather, do_scatter",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("apply_swiglu", [False, True])
@pytest.mark.parametrize("fused_quant", [False, True])
def test_op(
    m,
    n,
    k,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_swiglu,
    fused_quant,
    n_expts_tot,
    n_expts_act,
    act_dtype_str,
    hbm_swizzling,
    device="cuda",
):

    if get_arch() != "gfx950":
        pytest.skip("float8 x mx only supported on CDNA4")

    if "float8_e4m3fnuz" in act_dtype_str and get_arch() != "gfx942":
        pytest.skip("float8_e4m3fnuz only tested on AMD CDNA3 Platform")

    if hbm_swizzling:
        if get_arch() != "gfx950":
            pytest.skip(
                "Scale preshuffling on AMD GPU has not been emulated on non-CDNA4 arch yet."
            )
        if n % 32 != 0 or k % (32 * 8) != 0:
            pytest.skip(
                f"Shape {m}x{n}x{k} is not supported for scale swizzling on AMD GPU"
            )

    torch.manual_seed(0)

    weight_dtype_str = "mxfloat4_e2m1"
    weight_mxfp = weight_dtype_str.startswith("mx")
    if weight_mxfp:
        weight_dtype_str = weight_dtype_str[2:]
    act_mxfp8 = act_dtype_str.startswith("mx")
    if act_mxfp8:
        act_dtype_str = act_dtype_str[2:]

    weight_dtype = dtype_str_to_torch(weight_dtype_str)
    act_dtype = dtype_str_to_torch(act_dtype_str)
    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    x_tri, w_tri, bias_tri, gammas = init_compute_data(
        m,
        n,
        k,
        gindx,
        sindx,
        n_expts_tot,
        n_expts_act,
        torch.bfloat16 if act_mxfp8 else act_dtype,
        torch.bfloat16,
        has_y_gammas,
        device=device,
    )
    x_ref, w_ref, bias_ref = x_tri.clone(), w_tri.clone(), bias_tri.clone()

    # downcast to mxfp
    w_tri, w_scale_tri = downcast_to_mxfp(w_tri, weight_dtype, axis=1)
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)
    if hbm_swizzling:
        swizzle_mx_scale = "CDNA4_SCALE"
        w_scale_tri = swizzle_scales(w_scale_tri)
    else:
        swizzle_mx_scale = None

    if act_mxfp8:
        x_tri, x_mx_scales_tri = downcast_to_mxfp(x_tri, act_dtype, axis=-1)
        x_ref = upcast_from_mxfp(x_tri, x_mx_scales_tri, torch.bfloat16, axis=-1)
        x_static_scale = None
        out_dtype = torch.bfloat16
        maxtol = None
        rmstol = None
    else:
        x_mx_scales_tri = None
        x_static_scale = x_tri.abs().max().float() / 448.0
        x_tri = downcast_to_static_fp8(x_tri, x_static_scale)
        out_dtype = torch.float8_e4m3fn
        maxtol = 4e-1
        rmstol = 4e-2

    ref_y = moe_gemm_torch(
        x_ref, w_ref, bias_ref, rdata, gindx, sindx, gammas, apply_swiglu
    )
    if not act_mxfp8 and fused_quant:
        quant_static_scale = ref_y.abs().max().float() / 448.0
    else:
        quant_static_scale = None
    tri_y = moe_gemm_a8w4(
        x_tri,
        w_tri,
        x_mx_scales_tri,
        w_scale_tri,
        x_static_scale,
        quant_static_scale,
        bias_tri,
        rdata,
        gindx,
        sindx,
        gammas,
        swizzle_mx_scale,
        out_dtype,
        apply_swiglu,
    )
    if not act_mxfp8 and fused_quant:
        tri_y = (tri_y.float() * quant_static_scale).to(ref_y.dtype)
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)

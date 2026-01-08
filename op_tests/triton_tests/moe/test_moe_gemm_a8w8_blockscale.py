# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_matmul.py

from dataclasses import dataclass, fields
import pytest
import torch

# routing utilities
from aiter.ops.triton.moe.moe_routing.routing import routing

# matmul utilities
from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (
    moe_gemm_a8w8_blockscale,
    moe_gemm_torch,
)

# numerics utilities
from aiter.ops.triton.moe.quant_moe import (
    dequant_x_blockscale,
    dequant_w_blockscale,
)

# ---------------
# initialize data
# ---------------

# Default group_m, group_n, group_k
group_shape = (128, 128, 128)


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
    per_row_x_scale=False,
    is_x_blockscale=False,
    is_w_blockscale=False,
):
    torch.manual_seed(0)
    in_m = m * (n_expts_act if gindx is None else 1)
    shape_x = (in_m, k)
    x = (torch.randn(shape_x, dtype=torch.bfloat16, device=device) / 10).to(act_dtype)
    w = (torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device) / 10).to(
        weight_dtype
    )
    bias = torch.randn((n_expts_tot, n), dtype=torch.float32, device=device)
    if has_y_gammas:
        gamma = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gamma = None

    group_shape_m, group_shape_n, group_shape_k = group_shape
    scale_m = (in_m + group_shape_m - 1) // group_shape_m
    scale_n = (n + group_shape_n - 1) // group_shape_n
    scale_k = (k + group_shape_k - 1) // group_shape_k
    x_scale = None
    w_scale = None
    x_static_scale = None
    w_static_scale = None

    if is_x_blockscale:
        if per_row_x_scale:
            x_scale = torch.randn((in_m, scale_k), dtype=torch.float32, device="cuda")
        else:
            x_scale = torch.randn(
                (scale_m, scale_k), dtype=torch.float32, device="cuda"
            )
    else:
        x_static_scale = x.float().abs().max() / 448.0

    if is_w_blockscale:
        w_scale = torch.randn(
            (n_expts_tot, scale_k, scale_n), dtype=torch.float32, device="cuda"
        )
    else:
        w_static_scale = w.float().abs().max() / 448.0
    return x, x_scale, x_static_scale, w, w_scale, w_static_scale, bias, gamma


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
    n_expts_tot: int = 1
    n_expts_act: int = 1
    is_x_blockscale: bool = True
    is_w_blockscale: bool = True
    per_row_x_scale: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # 2D blockscale
            Case(16, 4096, 7168, 256, 8, True, True, per_row_x_scale=False),
            Case(256, 7168, 2048, 256, 8, True, True, per_row_x_scale=False),
            Case(1024, 4096, 7168, 256, 8, True, True, per_row_x_scale=False),
            Case(8192, 7168, 2048, 256, 8, True, True, per_row_x_scale=False),
            Case(4, 300, 300, 8, 4, True, True, per_row_x_scale=False),
            Case(4, 256, 256, 8, 2, True, True, per_row_x_scale=False),
            # Per-row x scale
            Case(16, 4096, 7168, 256, 8, True, True, per_row_x_scale=True),
            Case(256, 7168, 2048, 256, 8, True, True, per_row_x_scale=True),
            Case(1024, 4096, 7168, 256, 8, True, True, per_row_x_scale=True),
            Case(8192, 7168, 2048, 256, 8, True, True, per_row_x_scale=True),
            Case(4, 300, 300, 8, 4, True, True, per_row_x_scale=True),
            Case(4, 256, 256, 8, 2, True, True, per_row_x_scale=True),
            # Mixed quantization 2D blockscale
            Case(16, 512, 1024, 128, 4, False, False, per_row_x_scale=False),
            Case(256, 1024, 2048, 128, 8, True, True, per_row_x_scale=False),
            Case(1024, 7168, 4096, 256, 4, True, True, per_row_x_scale=False),
            Case(2048, 4096, 7168, 256, 8, True, True, per_row_x_scale=False),
            Case(4, 300, 300, 8, 2, False, True, per_row_x_scale=False),
            Case(4, 300, 300, 8, 4, True, False, per_row_x_scale=False),
            # Mixed quantization 2D blockscale
            Case(16, 512, 1024, 128, 4, False, False, per_row_x_scale=True),
            Case(256, 1024, 2048, 128, 8, True, True, per_row_x_scale=True),
            Case(1024, 7168, 4096, 256, 4, True, True, per_row_x_scale=True),
            Case(2048, 4096, 7168, 256, 8, True, True, per_row_x_scale=True),
            Case(4, 300, 300, 8, 2, False, True, per_row_x_scale=True),
            Case(4, 300, 300, 8, 4, True, False, per_row_x_scale=True),
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
    is_x_blockscale,
    is_w_blockscale,
    per_row_x_scale,
    device="cuda",
):
    torch.manual_seed(0)

    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    (
        x_tri,
        x_scales_tri,
        x_static_scale,
        w_tri,
        w_scales_tri,
        w_static_scale,
        bias_tri,
        gammas,
    ) = init_compute_data(
        m,
        n,
        k,
        gindx,
        sindx,
        n_expts_tot,
        n_expts_act,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        has_y_gammas,
        device=device,
        per_row_x_scale=per_row_x_scale,
        is_x_blockscale=is_x_blockscale,
        is_w_blockscale=is_w_blockscale,
    )
    x_ref, w_ref, bias_ref = x_tri.clone(), w_tri.clone(), bias_tri.clone()

    # Assume that x and w are quantized shapes
    if is_x_blockscale:
        x_ref = dequant_x_blockscale(
            x_tri, x_scales_tri, per_row_x_scale, group_shape
        ).to(torch.bfloat16)
    else:
        x_ref = (x_tri.float() * x_static_scale).to(torch.bfloat16)

    if is_w_blockscale:
        w_ref = dequant_w_blockscale(w_tri, w_scales_tri, group_shape).to(
            torch.bfloat16
        )
    else:
        w_ref = (w_tri.float() * w_static_scale).to(torch.bfloat16)
    ref_y = moe_gemm_torch(
        x_ref, w_ref, bias_ref, rdata, gindx, sindx, gammas, apply_swiglu
    )

    out_dtype = torch.bfloat16
    quant_static_scale = None
    if fused_quant and not is_x_blockscale:
        quant_static_scale = ref_y.abs().max().float() / 448.0
        maxtol = 4e-1
        rmstol = 4e-2
    elif is_x_blockscale and is_w_blockscale:
        maxtol = 2e-1
        rmstol = 4e-2
    elif is_x_blockscale or is_w_blockscale:
        maxtol = 3e-1
        rmstol = 4e-2
    else:
        maxtol = None
        rmstol = None
    tri_y = moe_gemm_a8w8_blockscale(
        x_tri,
        w_tri,
        x_scales_tri,
        w_scales_tri,
        x_static_scale,
        w_static_scale,
        quant_static_scale,
        bias_tri,
        rdata,
        gindx,
        sindx,
        gammas,
        out_dtype,
        apply_swiglu,
        per_row_x_scale=per_row_x_scale,
    )
    if not is_x_blockscale and fused_quant:
        tri_y = (tri_y.float() * quant_static_scale).to(ref_y.dtype)
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)

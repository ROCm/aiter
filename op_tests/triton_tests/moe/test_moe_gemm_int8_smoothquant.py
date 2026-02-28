# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_matmul.py

from dataclasses import dataclass, fields
import pytest
import torch

# Routing utilities
from aiter.ops.triton.moe.moe_routing.routing import routing

# SmoothQuant quantization utilities
from aiter.ops.triton.moe.quant_moe import (
    smoothquant_quantize,
    quantize_weights_int8,
)

# SmoothQuant MoE utilities
from aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant import (
    moe_gemm_int8_smoothquant,
    moe_gemm_smoothquant_torch,
    preshuffle_weights,
)

# ---------------
# Initialize data
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
    """Initialize computation data for MoE."""
    torch.manual_seed(0)
    in_m = m * (n_expts_act if gindx is None else 1)
    shape_x = (in_m, k)
    x = alloc_rand(shape_x, device=device, dtype=act_dtype)
    w = alloc_rand((n_expts_tot, k, n), device=device, dtype=weight_dtype)
    bias = alloc_rand((n_expts_tot, n), device=device, dtype=torch.float32)
    smooth_scale = alloc_rand((k,), device=device, dtype=torch.float32).abs() + 0.1
    if has_y_gammas:
        gamma = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gamma = None

    return x, w, bias, smooth_scale, gamma


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
    preshuffled: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            Case(16, 256, 256, 8, 2),
            Case(64, 512, 512, 8, 2),
            Case(128, 1024, 512, 8, 4),
            Case(256, 2048, 1024, 16, 4),
            Case(256, 2048, 2048, 32, 2, preshuffled=True),
            Case(512, 4096, 2048, 128, 8),
            Case(1024, 7168, 4096, 64, 8),
            Case(2048, 4096, 7168, 128, 8),
            Case(300, 400, 400, 8, 2),
            Case(16, 2560, 4096, 128, 6),
            Case(32, 2560, 4096, 128, 6, preshuffled=True),
            Case(128, 2560, 4096, 128, 6),
            Case(512, 2560, 4096, 128, 6),
            Case(2048, 2560, 4096, 128, 6),
            Case(16, 4096, 1280, 128, 6),
            Case(16, 4096, 1280, 128, 6, preshuffled=True),
            Case(128, 4096, 1280, 128, 6),
            Case(512, 4096, 1280, 128, 6),
            Case(2048, 4096, 1280, 128, 6),
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
@pytest.mark.parametrize("apply_activation", [False, True])
def test_op(
    m,
    n,
    k,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_activation,
    n_expts_tot,
    n_expts_act,
    preshuffled,
    device="cuda",
):
    torch.manual_seed(0)

    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    x_tri, w_tri, bias_tri, smooth_scale, gammas = init_compute_data(
        m,
        n,
        k,
        gindx,
        sindx,
        n_expts_tot,
        n_expts_act,
        torch.bfloat16,
        torch.bfloat16,
        has_y_gammas,
        device=device,
    )
    bias_ref = bias_tri.clone()

    x_int8_tri, x_scale_tri = smoothquant_quantize(x_tri, smooth_scale)
    x_int8_ref, x_scale_ref = x_int8_tri.clone(), x_scale_tri.clone()

    w_int8_tri, w_scale_tri = quantize_weights_int8(w_tri)
    w_int8_ref = w_int8_tri.clone()
    if preshuffled:
        w_int8_tri = preshuffle_weights(w_int8_tri)

    out_dtype = torch.bfloat16
    maxtol = 4e-2
    rmstol = None

    ref_y = moe_gemm_smoothquant_torch(
        x_int8_ref,
        x_scale_ref,
        w_int8_ref,
        w_scale_tri,
        bias_ref,
        rdata,
        gindx,
        sindx,
        gammas,
        apply_activation=apply_activation,
        add_residual=apply_activation,
    )

    tri_y = moe_gemm_int8_smoothquant(
        x_int8_tri,
        w_int8_tri,
        x_scale_tri,
        w_scale_tri,
        bias_tri,
        rdata,
        gindx,
        sindx,
        gammas,
        preshuffled,
        out_dtype,
        apply_activation=apply_activation,
        add_residual=apply_activation,
    )
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)

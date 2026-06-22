# gpt-oss MoE a8w4 (fp8 activation / mxfp4 weight) correctness on the gfx1250
# gluon path, for the small-batch decode regime (M = 1, 4, 64).
#
# Why this test exists separately from test_moe_gemm_a8w4.py:
#   - That file SKIPS the gpt-oss regime on gfx1250 (n_expts_tot > 32, n/k > 1024),
#     so the 128-expert gate_up+swiglu numerics are never exercised there.
#   - That file also feeds fp8-exact activations (alloc_rand emits powers of 2),
#     which makes the static-fp8 activation quant lossless and hides any real
#     numerical behaviour through the swiglu nonlinearity.
#
# Here we use realistic randn activations and compare against a SAME-QUANT
# reference: the torch reference consumes the identical fp8-rounded activations
# and mxfp4-rounded weights the kernel sees. That isolates kernel correctness
# from input quantization error (which would otherwise dominate the swiglu path
# and is not a kernel property).

import pytest
import torch

from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
    moe_gemm_a8w4,
    moe_gemm_torch,
    swizzle_scales_gfx1250,
)
from aiter.ops.triton.moe.quant_moe import (
    downcast_to_static_fp8,
    downcast_to_mxfp,
    upcast_from_mxfp,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

# gpt-oss proxy: tile-friendly 3072 (k % 256 == 0 so HBM scale-swizzling applies,
# unlike the real 2880 which can't be swizzled). 128 experts, top-4.
HIDDEN = 3072
INTER = 3072
N_EXPTS_TOT = 128
N_EXPTS_ACT = 4

# RMS is the robust aggregate signal. The swiglu (gate_up) path produces isolated
# large-relative-error elements at the gelu/clamp zero-crossings where fp8 rounding
# flips the sign of a near-zero output; those are expected and grow with M, so the
# max bound is a loose breakage guard there while RMS stays tight.
RMS_TOL = 4e-2
MAX_TOL_LINEAR = 4e-1
MAX_TOL_SWIGLU = 1.0


def _norm_err(ref, tri):
    """Normalized relative error, matching assert_close in test_moe_gemm_a8w4.py."""
    ref = ref.float()
    tri = tri.float()
    eps = 1e-30
    mult = 1.0 / (ref.abs().max() + eps)
    refn, trin = ref * mult, tri * mult
    ref_rms = refn.pow(2).mean().sqrt() + eps
    rel = (refn - trin).abs() / torch.maximum(ref_rms, refn.abs())
    return rel.max().item(), rel.pow(2).mean().sqrt().item()


# (n, k, do_gather, do_scatter, apply_swiglu)
_GEMMS = {
    "gemm1_gate_up": (2 * INTER, HIDDEN, True, False, True),
    "gemm2_down": (HIDDEN, INTER, False, True, False),
}


@pytest.mark.parametrize("m", [1, 4, 64])
@pytest.mark.parametrize("gemm", list(_GEMMS))
def test_gptoss_a8w4_same_quant(m, gemm, device="cuda"):
    if get_arch() != "gfx1250":
        pytest.skip("gluon a8w4 gpt-oss path is gfx1250-only")

    n, k, do_gather, do_scatter, apply_swiglu = _GEMMS[gemm]
    torch.manual_seed(0)

    logits = torch.randn((m, N_EXPTS_TOT), dtype=torch.float16, device=device)
    rdata, gather_idx, scatter_idx = routing(logits, N_EXPTS_ACT)
    rdata.gate_scal = None
    gindx = gather_idx if do_gather else None
    sindx = scatter_idx if do_scatter else None

    in_m = m * (N_EXPTS_ACT if gindx is None else 1)
    x = torch.randn((in_m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((N_EXPTS_TOT, k, n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((N_EXPTS_TOT, n), device=device, dtype=torch.float32)

    # mxfp4 weights — same quant for kernel and reference.
    w_q, w_scale = downcast_to_mxfp(w, torch.uint8, axis=1)
    w_deq = upcast_from_mxfp(w_q, w_scale, torch.bfloat16, axis=1)
    w_scale_sw = swizzle_scales_gfx1250(w_scale)

    # static-scaled fp8 activations — same quant for kernel and reference.
    x_scale = x.abs().max().float() / 448.0
    x_q = downcast_to_static_fp8(x, x_scale)
    x_deq = (x_q.float() * x_scale).to(torch.bfloat16)

    ref = moe_gemm_torch(x_deq, w_deq, bias, rdata, gindx, sindx, None, apply_swiglu)
    out = moe_gemm_a8w4(
        x_q,
        w_q,
        None,
        w_scale_sw,
        x_scale,
        None,
        bias,
        rdata,
        gindx,
        sindx,
        None,
        "GFX1250_SCALE",
        torch.bfloat16,
        apply_swiglu,
    )

    max_err, rms_err = _norm_err(ref, out)
    max_tol = MAX_TOL_SWIGLU if apply_swiglu else MAX_TOL_LINEAR
    assert rms_err <= RMS_TOL, f"{gemm} M={m}: rms {rms_err:.4f} > {RMS_TOL} (max {max_err:.4f})"
    assert max_err <= max_tol, f"{gemm} M={m}: max {max_err:.4f} > {max_tol} (rms {rms_err:.4f})"

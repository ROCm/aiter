"""Standalone numerical check: moe_gemm_a16w4 with GFX1250_SCALE vs torch ref.

Mirrors test_moe_gemm_a16w4.py::test_op but forces the gfx1250 native scale
preshuffle (shuffle_scale_moe(arch='gfx1250') -> 'GFX1250_SCALE') to validate the
a16w4 kernel's new GFX1250_SCALE unswizzle branch on real gfx1250 hardware.
"""
import sys
import torch

from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4, moe_gemm_torch
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
import aiter.ops.triton.utils._triton.arch_info as arch_info


def alloc_rand(shape, device, dtype):
    if dtype.itemsize == 1:
        return 2 ** -(torch.randint(4, 8, shape, device=device, dtype=torch.bfloat16))
    return torch.randn(shape, device=device, dtype=dtype)


def run(m, n, k, n_expts_tot, n_expts_act, apply_swiglu, arch, label, device="cuda"):
    torch.manual_seed(0)
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    rdata, gindx, sindx = routing(logits, n_expts_act)
    rdata.gate_scal = None

    in_m = m * n_expts_act  # gindx is None -> no gather
    x = alloc_rand((in_m, k), device, torch.bfloat16)
    w = alloc_rand((n_expts_tot, k, n), device, torch.bfloat16)
    bias = alloc_rand((n_expts_tot, n), device, torch.float32)

    x_ref, w_ref, bias_ref = x.clone(), w.clone(), bias.clone()

    weight_dtype = torch.uint8  # mxfp4_e2m1 packed
    from aiter.ops.triton.utils.types import str_to_torch_dtype
    w_tri, w_scale_tri = downcast_to_mxfp(w, str_to_torch_dtype["mxfp4_e2m1"], axis=1)
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)

    if arch is None:
        swizzle = None
        w_scale_use = w_scale_tri
    else:
        w_scale_use = shuffle_scale_moe(
            w_scale_tri, arch=arch, preshuffle_factor=32, scale_kwidth=8
        )
        swizzle = label

    ref_y = moe_gemm_torch(x_ref, w_ref, bias_ref, rdata, None, None, None, apply_swiglu)
    tri_y = moe_gemm_a16w4(
        x, w_tri, None, w_scale_use, None, None, bias, rdata,
        None, None, None, swizzle, torch.bfloat16, apply_swiglu,
    )

    ref = ref_y.to(torch.float32)
    tri = tri_y.to(torch.float32)
    mult = 1.0 / (ref.abs().max() + 1e-30)
    refn, trin = ref * mult, tri * mult
    ref_rms = (refn.square().mean().sqrt()) + 1e-30
    rel = (refn - trin).abs() / torch.maximum(ref_rms, refn.abs())
    max_err = rel.max().item()
    rms_err = rel.square().mean().sqrt().item()
    ok = max_err <= 4e-1 and rms_err <= 4e-2
    print(f"[{label:14s} swiglu={int(apply_swiglu)}] max_err={max_err:.4f} "
          f"rms_err={rms_err:.4f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("arch:", arch_info.get_arch() if hasattr(arch_info, "get_arch") else "?")
    m, n, k, E, A = 64, 1024, 1024, 8, 4
    all_ok = True
    for sw in (False, True):
        all_ok &= run(m, n, k, E, A, sw, None, "None")
        all_ok &= run(m, n, k, E, A, sw, "gfx950", "CDNA4_SCALE")
        all_ok &= run(m, n, k, E, A, sw, "gfx1250", "GFX1250_SCALE")
    sys.exit(0 if all_ok else 1)

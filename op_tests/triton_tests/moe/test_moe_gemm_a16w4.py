# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_matmul.py

import dataclasses

import pytest
import torch

# backend selection for moe_gemm_a16w4 (triton vs gfx1250 gluon)
from aiter.ops.triton.utils._triton.arch_info import get_arch

# routing utilities
from aiter.ops.triton.moe.moe_routing.routing import routing

# matmul utilities
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    moe_gemm_a16w4,
    moe_gemm_torch,
)
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

# numerics utilities
from aiter.ops.triton.moe.quant_moe import (
    # downcast_to_static_fp8,
    downcast_to_mxfp,
    upcast_from_mxfp,
)

# target-specific utilities
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype

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

    # Compare reference values against obtained values.
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


# Test matrix: SHAPES x EXPERTS x SWIZZLE x apply_swiglu, plus gather/scatter,
# gammas and backend axes below.
SHAPES = [
    (4, 4, 8),
    (4, 1024, 3072),
    (16, 256, 256),
    (16, 1024, 1024),
    (300, 400, 800),
    (1000, 704, 800),
    (1024, 3072, 512),
    (4096, 256, 256),
    (4096, 3072, 3072),
    (4097, 1024, 1024),
    (8192, 3072, 3072),
    # MiniMax-M3 gate/up (n=k=6144) and down (n=6144, k=3072), decode + prefill M.
    (16, 6144, 6144),
    (32, 6144, 3072),
    (4096, 6144, 6144),
    (4096, 6144, 3072),
]
# MiniMax-M3 routing: 128 experts, top-4.
EXPERTS = [(128, 4)]
SWIZZLES = [None, "CDNA4_SCALE", "GFX1250_SCALE"]


@pytest.mark.parametrize("m, n, k", SHAPES)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", EXPERTS)
@pytest.mark.parametrize("hbm_swizzling", SWIZZLES)
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
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_op(
    m,
    n,
    k,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_swiglu,
    n_expts_tot,
    n_expts_act,
    hbm_swizzling,
    backend,
    golden_cpu,
    device="cuda",
):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    # reduce_grouped static_asserts NPAD >= 32, so N must be >= 32. Only the
    # degenerate (4, 4, 8) smoke shape violates this; skip its scatter cases.
    if do_scatter and n < 32:
        pytest.skip(f"scatter-combine (reduce_grouped) requires N >= 32, got N={n}")

    # Pin the backend via the config arg.
    backend_config = {"backend": backend}
    if backend == "gluon":
        # gluon a16w4 only dispatches on gfx1250; elsewhere it falls back to
        # Triton, making this run a duplicate.
        if get_arch() != "gfx1250":
            pytest.skip("gluon a16w4 backend is only available on gfx1250")
        # gluon a16w4 only supports compact (non-swizzled) e8m0 scales
        # (CDNA4_SCALE unsupported, GFX1250_SCALE hangs the ROCm loader), so
        # any swizzled scale raises. Skip those cases.
        if hbm_swizzling:
            pytest.skip(
                f"gluon a16w4 backend does not support swizzled scales ({hbm_swizzling})"
            )

    if hbm_swizzling:
        if not arch_info.is_mx_scale_preshuffling_avail():
            pytest.skip(
                "Scale preshuffling on AMD GPU has not been emulated on non-CDNA4 arch yet."
            )
        if n % 32 != 0 or k % (32 * 8) != 0:
            pytest.skip(
                f"Shape {m}x{n}x{k} is not supported for scale swizzling on AMD GPU"
            )

    # Known bug: triton CDNA4_SCALE + swiglu is numerically wrong for the MiniMax
    # gate/up prefill shape (block_m=128, N=K=6144) on gfx1250 (RMS ~0.016 vs
    # ~5e-6); None/GFX1250_SCALE and this shape without swiglu are fine. Drop this
    # xfail once the CDNA4 unswizzle is fixed for that config.
    if (
        hbm_swizzling == "CDNA4_SCALE"
        and (m, n, k) == (4096, 6144, 6144)
        and apply_swiglu
    ):
        pytest.xfail(
            "triton CDNA4_SCALE wrong on gfx1250 prefill (block_m=128, N=K=6144) "
            "with swiglu"
        )

    torch.manual_seed(0)

    weight_dtype_str = "mxfp4_e2m1"
    weight_dtype = str_to_torch_dtype[weight_dtype_str]

    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )

    # x: (m, k)
    # w: (num_expts_tot, k, n)
    # bias: (num_expts_tot, n)
    # gammas: (m*num_expts_act)
    x_tri, w_tri, bias_tri, gammas = init_compute_data(
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
    x_ref, w_ref, bias_ref = x_tri.clone(), w_tri.clone(), bias_tri.clone()

    # downcast to mxfp
    w_tri, w_scale_tri = downcast_to_mxfp(w_tri, weight_dtype, axis=1)
    # --golden-cpu computes the golden on CPU (frees GPU memory for the kernel; the
    # per-expert upcast still runs on GPU then moves to the golden device).
    # Per-expert upcast is bit-identical to a bulk upcast and avoids its
    # int32-indexing overflow on > 2**31-element weights (MiniMax-M3, 128 experts).
    golden_dev = "cpu" if golden_cpu else "cuda"
    w_ref = torch.stack(
        [
            upcast_from_mxfp(w_tri[e], w_scale_tri[e], torch.bfloat16, axis=0).to(
                golden_dev
            )
            for e in range(w_tri.shape[0])
        ]
    )
    if hbm_swizzling:
        # hbm_swizzling names the target layout: CDNA4_SCALE via gfx950 shuffle,
        # GFX1250_SCALE via gfx1250 (both preshuffle_factor=32, scale_kwidth=8).
        # return_layout hands back the canonical label so the kernel's
        # SWIZZLE_MX_SCALE matches the layout actually applied.
        shuffle_arch = "gfx950" if hbm_swizzling == "CDNA4_SCALE" else "gfx1250"
        w_scale_tri, swizzle_mx_scale = shuffle_scale_moe(
            w_scale_tri,
            arch=shuffle_arch,
            preshuffle_factor=32,
            scale_kwidth=8,
            return_layout=True,
        )
        assert swizzle_mx_scale == hbm_swizzling, (swizzle_mx_scale, hbm_swizzling)
    else:
        swizzle_mx_scale = None

    x_mx_scales_tri = None
    out_dtype = torch.bfloat16
    x_static_scale = None
    quant_static_scale = None
    maxtol = 4e-1
    rmstol = 4e-2

    if golden_dev == "cpu":
        # moe_gemm_torch follows x.device; move all golden inputs to CPU (routing
        # histogram included) and run the reference matmul on CPU.
        rdata_g = dataclasses.replace(
            rdata,
            expt_hist=None if rdata.expt_hist is None else rdata.expt_hist.cpu(),
        )
        ref_y = moe_gemm_torch(
            x_ref.cpu(),
            w_ref,
            bias_ref.cpu(),
            rdata_g,
            None if gindx is None else gindx.cpu(),
            None if sindx is None else sindx.cpu(),
            None if gammas is None else gammas.cpu(),
            apply_swiglu,
        )
    else:
        ref_y = moe_gemm_torch(
            x_ref, w_ref, bias_ref, rdata, gindx, sindx, gammas, apply_swiglu
        )

    tri_y = moe_gemm_a16w4(
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
        config=backend_config,
    )
    assert_close(ref_y.to(tri_y.device), tri_y, maxtol=maxtol, rmstol=rmstol)


def test_gluon_stage_wrappers_signature_match():
    """The stage1/stage2/stage3 gluon entry points are thin wrappers that must
    forward the *exact* signature of _moe_gemm_a16w4_gluon_impl. Because each is
    a full hand-written copy of the ~48-arg signature, a drift (added, removed,
    renamed or reordered param) would silently mis-forward one pipeline. Guard
    against that here so the duplication stays honest."""
    if get_arch() != "gfx1250":
        pytest.skip("gluon a16w4 kernels are only built on gfx1250")
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w4 import (
        _moe_gemm_a16w4_gluon_impl,
        _moe_gemm_a16w4_gluon_stage1,
        _moe_gemm_a16w4_gluon_stage2,
        _moe_gemm_a16w4_gluon_stage3,
    )

    ref = _moe_gemm_a16w4_gluon_impl.arg_names
    for wrapper in (
        _moe_gemm_a16w4_gluon_stage1,
        _moe_gemm_a16w4_gluon_stage2,
        _moe_gemm_a16w4_gluon_stage3,
    ):
        assert wrapper.arg_names == ref, (
            f"{wrapper.__name__} signature drifted from "
            f"_moe_gemm_a16w4_gluon_impl:\n  wrapper={wrapper.arg_names}\n  impl={ref}"
        )

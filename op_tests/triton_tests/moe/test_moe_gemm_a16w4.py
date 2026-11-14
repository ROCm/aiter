# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_matmul.py

import os
from dataclasses import dataclass, fields

import pytest
import torch

# matmul utilities
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    moe_gemm_a16w4,
    moe_gemm_torch,
)

# routing utilities
from aiter.ops.triton.moe.moe_routing.routing import routing

# numerics utilities
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

# target-specific utilities
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.triton_tests.moe.moe_test_utils import assert_close
from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp

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
    x = alloc_rand(shape_x, device=device, dtype=act_dtype)  # row-major
    w = alloc_rand((n_expts_tot, k, n), device=device, dtype=weight_dtype)  # row-major
    bias = alloc_rand((n_expts_tot, n), device=device, dtype=torch.float32)
    if has_y_gammas:
        gamma = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gamma = None
    return x, w, bias, gamma


# ---------------
# unit tests
# ---------------


_TUNED_ARCH = "gfx942"


@dataclass
class Case:
    m: int
    n: int
    k: int
    n_expts_tot: int = 1
    n_expts_act: int = 1
    hbm_swizzling: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            Case(4, 4, 8, 2, 1),
            Case(4, 4, 8, 8, 2),
            Case(4, 4, 8, 128, 4),
            Case(4, 32, 64, 128, 4),
            Case(4, 1024, 3072, 128, 4),
            Case(32, 6144, 3072, 128, 4),
            Case(16, 1024, 1024, 128, 4),
            Case(16, 128, 128, 2, 1),
            Case(16, 256, 256, 128, 4),
            Case(4096, 256, 256, 128, 4),
            Case(1024, 3072, 512, 128, 4),
            Case(4096, 3072, 3072, 128, 4),
            Case(8192, 3072, 3072, 128, 4),
            Case(300, 400, 800, 8, 4),
            Case(1000, 704, 800, 8, 2),
            Case(4097, 1024, 1024, 128, 4),
            # gfx942 tuned table: m=256 -> block_m=128, m=128 -> block_m=64
            Case(256, 1536, 5120, 2, 1),
            Case(128, 1536, 5120, 2, 1),
            Case(16, 32, 256, 2, 1, hbm_swizzling=True),
            Case(16, 256, 256, 8, 4, hbm_swizzling=True),
            Case(32, 6144, 3072, 128, 4, hbm_swizzling=True),
            Case(32, 6144, 3072, 8, 4, hbm_swizzling=True),
            Case(64, 512, 4096, 256, 6, hbm_swizzling=True),
            Case(16, 1024, 1024, 128, 4, hbm_swizzling=True),
            Case(16, 1024, 1024, 2, 1, hbm_swizzling=True),
            Case(16, 256, 256, 128, 4, hbm_swizzling=True),
            Case(1024, 3072, 512, 128, 4, hbm_swizzling=True),
            Case(4096, 256, 256, 128, 4, hbm_swizzling=True),
            Case(4097, 1024, 1024, 128, 4, hbm_swizzling=True),
            Case(8192, 3072, 3072, 128, 4, hbm_swizzling=True),
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
@pytest.mark.parametrize("backend", [None, "gluon", "triton"])
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
    device="cuda",
):

    if int(os.environ.get("AITER_IN_FFM_AM", "0")) == 1 and (
        m > 1024
        or n > 1024
        or k > 1024
        or n_expts_tot > 128
        or (m >= 1024 or n >= 1024 or k >= 1024 and n_expts_tot >= 128)
    ):
        pytest.skip("Test will take too long on FFM")

    if not (arch_info.is_fp4_avail() or arch_info.get_arch() == _TUNED_ARCH):
        pytest.skip("MXFP4 not supported on this architecture")

    if hbm_swizzling:
        if not arch_info.is_mx_scale_preshuffling_avail():
            pytest.skip(
                "Scale preshuffling on AMD GPU has not been emulated on non-CDNA4 arch yet."
            )
        if n % 32 != 0 or k % (32 * 8) != 0:
            pytest.skip(
                f"Shape {m}x{n}x{k} is not supported for scale swizzling on AMD GPU"
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
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)
    if hbm_swizzling:
        w_scale_tri, swizzle_mx_scale = shuffle_scale_moe(
            w_scale_tri, preshuffle_factor=32, scale_kwidth=8, return_layout=True
        )
    else:
        swizzle_mx_scale = None

    x_mx_scales_tri = None
    out_dtype = torch.bfloat16
    x_static_scale = None
    quant_static_scale = None
    maxtol = 4e-1
    rmstol = 4e-2

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
        backend=backend,
    )
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)


@pytest.mark.parametrize("m, n, k, n_expts_tot, n_expts_act", [(192, 1536, 5120, 384, 6), (24, 5120, 2304, 48, 6)])
def test_gfx942_gluon_fp16_operand_matches_bf16(m, n, k, n_expts_tot, n_expts_act, monkeypatch):
    """gfx942 FP16_OPERAND path (e8m0 scale folded into the fp16 byte lookup) against the bf16 path on the same
    inputs: every fp4 code, scales across the whole exact window [114, 140], zero-padding columns (zero weights with
    e8m0 0, as vLLM pads), and activations spanning the fp16 range."""
    if arch_info.get_arch() != "gfx942":
        pytest.skip("gfx942 Gluon a16w4 kernel only")
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as moe_mod

    device = "cuda"
    torch.manual_seed(0)
    m, rdata, gindx, sindx = init_routing_data(m, n_expts_tot, n_expts_act, True, False, device=device)
    x = (torch.randn((m, k), device=device) * 2.0 ** torch.randint(-12, 8, (m, k), device=device)).to(torch.bfloat16)
    w_packed = torch.randint(0, 256, (n_expts_tot, n, k // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randint(114, 141, (n_expts_tot, n, k // 32), dtype=torch.uint8, device=device)
    w_packed[:, -32:] = 0
    w_scale[:, -32:] = 0
    w_tri, s_tri = w_packed.transpose(1, 2), w_scale.transpose(1, 2)
    ys = {}
    for mode in ("0", "1"):
        monkeypatch.setattr(moe_mod, "_GFX942_FP16", mode)
        ys[mode] = moe_gemm_a16w4(
            x, w_tri, None, s_tri, None, None, None, rdata, gindx, None, None, None,
            torch.bfloat16, False, backend="gluon",
        )
    # same exact products and f32 accumulation (the bf16 path is checked against the reference by test_op); the
    # only operand difference is activations below 2^-17, which fp16 rounds: last-bit output differences at most
    atol = 2.0**-20 * ys["0"].abs().max().item()
    torch.testing.assert_close(ys["1"], ys["0"], rtol=2.0**-7, atol=atol)
    assert (ys["1"] == ys["0"]).float().mean().item() > 0.99
    assert (ys["1"][:, -32:] == 0).all()
    # |x| > 65504 saturates in the fp16 convert instead of producing inf
    x[: m // 2, ::7] = 1.0e5
    y_big = moe_gemm_a16w4(
        x, w_tri, None, s_tri, None, None, None, rdata, gindx, None, None, None, torch.bfloat16, False, backend="gluon"
    )
    assert torch.isfinite(y_big).all()


def test_gfx942_gluon_fp16_operand_scale_check(monkeypatch):
    """auto mode: the scale-range verdict is per scale storage and is redone for a new buffer at the same address."""
    if arch_info.get_arch() != "gfx942":
        pytest.skip("gfx942 Gluon a16w4 kernel only")
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as moe_mod

    monkeypatch.setattr(moe_mod, "_GFX942_FP16", "auto")
    monkeypatch.setattr(moe_mod, "_gfx942_fp16_scales_ok", {})
    for lo, hi, ok in ((114, 140, True), (113, 140, False), (114, 141, False), (120, 125, True)):
        s = torch.randint(lo, hi + 1, (8, 64, 160), dtype=torch.uint8, device="cuda")
        s[0, 0, 0], s[0, 0, 1], s[0, 1, 0] = lo, hi, 0
        assert moe_mod._gfx942_fp16_operand(s.transpose(1, 2)) is ok, (lo, hi)
        assert moe_mod._gfx942_fp16_operand(s.transpose(1, 2)) is ok, (lo, hi)  # cached
        del s


def test_gfx942_gluon_fp16_operand_graph_capture(monkeypatch):
    """vLLM order: a large-M eager call (the profile run, block_m > 32) primes the scale check, then a decode-size
    call is captured with no eager warmup at that size. The captured call must take the fp16 path from the cache
    (no host sync under capture) and replay must match eager."""
    if arch_info.get_arch() != "gfx942":
        pytest.skip("gfx942 Gluon a16w4 kernel only")
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as moe_mod

    monkeypatch.setattr(moe_mod, "_GFX942_FP16", "auto")
    real, seen = moe_mod._gfx942_fp16_operand, []

    def spy(scales):
        r = real(scales)
        seen.append((torch.cuda.is_current_stream_capturing(), r))
        return r

    monkeypatch.setattr(moe_mod, "_gfx942_fp16_operand", spy)
    device = "cuda"
    torch.manual_seed(0)
    n, k, n_expts_tot, n_expts_act = 1536, 5120, 384, 6
    w_packed = torch.randint(0, 256, (n_expts_tot, n, k // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randint(118, 129, (n_expts_tot, n, k // 32), dtype=torch.uint8, device=device)
    w_tri, s_tri = w_packed.transpose(1, 2), w_scale.transpose(1, 2)

    def make_call(m):
        m, rdata, gindx, _ = init_routing_data(m, n_expts_tot, n_expts_act, True, False, device=device)
        x = (torch.randn((m, k), device=device) * 0.1).to(torch.bfloat16)
        return lambda: moe_gemm_a16w4(
            x, w_tri, None, s_tri, None, None, None, rdata, gindx, None, None, None, torch.bfloat16, False, backend="gluon"
        )

    decode = make_call(192)
    y_eager = decode()  # compile outside capture
    monkeypatch.setattr(moe_mod, "_gfx942_fp16_scales_ok", {})
    make_call(8192)()  # profile-run size
    seen.clear()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_graph = decode()
    g.replay()
    torch.cuda.synchronize()
    assert seen == [(True, True)], seen
    assert torch.equal(y_graph, y_eager)


@pytest.mark.parametrize("m", [24, 256, 4096])
@pytest.mark.parametrize("n, k", [(1536, 5120), (5120, 768), (5120, 384)])
@pytest.mark.parametrize("fp16_mode", ["0", "auto"])
def test_gfx942_gluon_strip16_matches_column_major(m, n, k, fp16_mode, monkeypatch):
    """w_layout="GFX942_STRIP16": pack/unpack round trip, and the record layout gives a bit-identical output to the
    column-major layout (same registers, K grouping and MFMA order) at decode and prefill block sizes, on the bf16
    and the fp16 operand paths."""
    if arch_info.get_arch() != "gfx942":
        pytest.skip("gfx942 Gluon a16w4 kernel only")
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as moe_mod
    from aiter.ops.triton._gluon_kernels.gfx942.moe.moe_op_gemm_a16w4 import (
        W_LAYOUT_STRIP16,
        gfx942_strip16_pack,
        gfx942_strip16_unpack,
    )

    monkeypatch.setattr(moe_mod, "_GFX942_FP16", fp16_mode)
    device = "cuda"
    torch.manual_seed(0)
    n_expts_tot, n_expts_act = 64, 6
    m, rdata, gindx, sindx = init_routing_data(m, n_expts_tot, n_expts_act, True, False, device=device)
    x = (torch.randn((m, k), device=device) * 0.5).to(torch.bfloat16)
    w_packed = torch.randint(0, 256, (n_expts_tot, n, k // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randint(114, 141, (n_expts_tot, n, k // 32), dtype=torch.uint8, device=device)
    w_tri, s_tri = w_packed.transpose(1, 2), w_scale.transpose(1, 2)

    rec = gfx942_strip16_pack(w_tri, s_tri)
    w_back, s_back = gfx942_strip16_unpack(rec)
    assert torch.equal(w_back, w_tri) and torch.equal(s_back, s_tri)

    y_col = moe_gemm_a16w4(
        x, w_tri, None, s_tri, None, None, None, rdata, gindx, None, None, None, torch.bfloat16, False, backend="gluon"
    )
    y_rec = moe_gemm_a16w4(
        x, rec, None, s_tri, None, None, None, rdata, gindx, None, None, None, torch.bfloat16, False,
        backend="gluon", w_layout=W_LAYOUT_STRIP16,
    )
    assert torch.equal(y_rec, y_col)


def _ep_setup(
    m, n, k, n_expts_tot, n_expts_act, ep_size, ep_rank, do_gather, do_scatter
):
    device = "cuda"
    torch.manual_seed(0)
    weight_dtype = str_to_torch_dtype["mxfp4_e2m1"]
    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    x, w, bias, _ = init_compute_data(
        m,
        n,
        k,
        gindx,
        sindx,
        n_expts_tot,
        n_expts_act,
        torch.bfloat16,
        torch.bfloat16,
        False,
        device=device,
    )
    w_tri, w_scale_tri = downcast_to_mxfp(w, weight_dtype, axis=1)
    n_local = n_expts_tot // ep_size
    lo, hi = ep_rank * n_local, (ep_rank + 1) * n_local
    expert_map = torch.full((n_expts_tot,), -1, dtype=torch.int32, device=device)
    expert_map[lo:hi] = torch.arange(n_local, dtype=torch.int32, device=device)
    gate_valid = (expert_map[rdata.topk_ids.long()] >= 0).to(torch.int32).contiguous()
    return (
        x,
        w_tri,
        w_scale_tri,
        bias,
        rdata,
        gindx,
        sindx,
        lo,
        hi,
        expert_map,
        gate_valid,
    )


@pytest.mark.parametrize(
    "m, n, k, n_expts_tot, n_expts_act",
    [
        (16, 256, 256, 8, 2),
        (64, 512, 512, 128, 4),
        (300, 400, 800, 8, 4),
        (1024, 512, 512, 128, 4),
    ],
)
@pytest.mark.parametrize("ep_size, ep_rank", [(4, 0), (4, 2)])
@pytest.mark.parametrize(
    "do_gather, do_scatter", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("apply_swiglu", [False, True])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_op_expert_parallel(
    m,
    n,
    k,
    n_expts_tot,
    n_expts_act,
    ep_size,
    ep_rank,
    do_gather,
    do_scatter,
    apply_swiglu,
    backend,
):
    """Rank-local weights + global routing reproduce the local experts' output;
    non-local gates contribute exactly 0 to the top-k combine."""
    if not (arch_info.is_fp4_avail() or arch_info.get_arch() == _TUNED_ARCH):
        pytest.skip("MXFP4 not supported on this architecture")
    # expert_map on the gluon path is gfx942-only (other arches assert in the wrapper)
    if backend == "gluon" and arch_info.get_arch() != "gfx942":
        pytest.skip("gluon expert_map is only supported on gfx942")

    x, w_tri, w_scale_tri, bias, rdata, gindx, sindx, lo, hi, expert_map, gate_valid = (
        _ep_setup(
            m, n, k, n_expts_tot, n_expts_act, ep_size, ep_rank, do_gather, do_scatter
        )
    )

    # Reference on all global experts with non-local weights/bias zeroed.
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)
    bias_ref = bias.clone()
    is_nonlocal = torch.ones(n_expts_tot, dtype=torch.bool, device=x.device)
    is_nonlocal[lo:hi] = False
    w_ref[is_nonlocal] = 0
    bias_ref[is_nonlocal] = 0
    ref_y = moe_gemm_torch(
        x.clone(), w_ref, bias_ref, rdata, gindx, sindx, None, apply_swiglu
    )

    tri_y = moe_gemm_a16w4(
        x,
        w_tri[lo:hi],
        None,
        w_scale_tri[lo:hi],
        None,
        None,
        bias[lo:hi],
        rdata,
        gindx,
        sindx,
        None,
        None,
        torch.bfloat16,
        apply_swiglu,
        backend=backend,
        expert_map=expert_map,
        gate_valid=gate_valid,
    )

    if not do_scatter:
        # Sorted output: only local experts' rows are written.
        offs = rdata.expt_data.token_offs_raw
        r0, r1 = int(offs[lo]), int(offs[hi])
        ref_y, tri_y = ref_y[r0:r1], tri_y[r0:r1]
    assert_close(ref_y, tri_y, maxtol=4e-1, rmstol=4e-2)


def test_op_expert_parallel_rejects_bad_inputs():
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")
    x, w_tri, w_scale_tri, bias, rdata, gindx, sindx, lo, hi, expert_map, gate_valid = (
        _ep_setup(16, 256, 256, 8, 2, 4, 1, True, True)
    )
    args = (
        x,
        w_tri[lo:hi],
        None,
        w_scale_tri[lo:hi],
        None,
        None,
        bias[lo:hi],
        rdata,
        gindx,
        sindx,
    )
    with pytest.raises(AssertionError, match="requires gate_valid"):
        moe_gemm_a16w4(*args, backend="triton", expert_map=expert_map)
    with pytest.raises(AssertionError, match="n_expts_tot"):
        moe_gemm_a16w4(
            *args,
            backend="triton",
            expert_map=expert_map[:-1].contiguous(),
            gate_valid=gate_valid,
        )

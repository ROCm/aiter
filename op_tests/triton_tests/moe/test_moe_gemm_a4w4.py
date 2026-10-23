# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_matmul.py

from dataclasses import dataclass, fields

import pytest
import torch
import triton

from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

# matmul utilities
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
    moe_gemm_a4w4,
    moe_gemm_torch,
    mxfp4_quant,
)

# routing utilities
from aiter.ops.triton.moe.moe_routing.routing import (
    _compute_expt_data_internal,
    EpScatterGeometry,
    ep_sort_routing,
    ExptData,
    RoutingData,
    routing,
)
from aiter.ops.triton.moe.reduce import EpCombineScatter

# numerics utilities
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

# target-specific utilities
from aiter.ops.triton.utils._triton.arch_info import get_arch, is_fp4_avail
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view, shuffle_scale_moe
from op_tests.triton_tests.moe.moe_test_utils import assert_close
from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp


def preshuffle_moe_weight(w: torch.Tensor) -> torch.Tensor:
    """``(E, K, N)`` -> the gfx1250 WMMA TDM view ``(E, K*16, N//16)``.

    ``moe_shuffle_weight`` takes the ``(E, N, K)`` MoE weight orientation and
    returns the shuffled buffer in that same shape; ``moe_weight_decode_view``
    then reinterprets it (zero-copy) as the flattened view the kernel loads.
    """
    return moe_weight_decode_view(moe_shuffle_weight(w.transpose(-1, -2)))


def preshuffle_moe_wscale(s: torch.Tensor) -> torch.Tensor:
    """``(E, K//32, N)`` B-scale -> gfx1250 n32k4 layout, same orientation back.

    ``moe_shuffle_scale`` is the n32k4 tile (preshuffle 32, scale kwidth 4) and
    takes the ``(E, N, K//32)`` orientation, so transpose in and back out. This
    is the kwidth-4 counterpart of ``shuffle_scale_moe(..., scale_kwidth=4)``
    and must stay in step with ``SCALE_KWIDTH`` in the gfx1250 gluon kernels.
    """
    return moe_shuffle_scale(s.transpose(-1, -2)).transpose(-1, -2)


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
    hbm_swizzling: bool = False
    preshuffle_weights: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # TP1
            Case(16, 4096, 7168, 256, 8, hbm_swizzling=True),
            Case(1024, 7168, 2048, 256, 8, hbm_swizzling=True),
            Case(4096, 4096, 7168, 256, 8, hbm_swizzling=True),
            Case(8192, 7168, 2048, 256, 8, hbm_swizzling=True),
            # TP8
            Case(16, 512, 7168, 256, 8, hbm_swizzling=True),
            Case(1024, 7168, 256, 256, 8, hbm_swizzling=True),
            Case(4096, 512, 7168, 256, 8, hbm_swizzling=True),
            Case(8192, 7168, 256, 256, 8, hbm_swizzling=True),
            # Edges
            Case(1000, 704, 800, 8, 2),
            Case(300, 400, 800, 8, 4),
            Case(256, 1024, 1024, 8, 4, hbm_swizzling=True),
            Case(32, 6144, 3072, 128, 4, hbm_swizzling=True),
            Case(4096, 3072, 3072, 128, 4),
            Case(8192, 7168, 4096, 256, 8),
            # gfx1250 gluon preshuffled weights
            Case(16, 4096, 7168, 256, 8, hbm_swizzling=True, preshuffle_weights=True),
            Case(16, 512, 7168, 256, 8, hbm_swizzling=True, preshuffle_weights=True),
            Case(16, 1024, 1024, 128, 4, preshuffle_weights=True),
            Case(1024, 7168, 2048, 256, 8, hbm_swizzling=True, preshuffle_weights=True),
            Case(256, 1024, 1024, 8, 4, preshuffle_weights=True),
            Case(16, 1536, 7168, 256, 8, hbm_swizzling=True, preshuffle_weights=True),
            Case(16, 7168, 768, 256, 8, hbm_swizzling=True, preshuffle_weights=True),
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
@pytest.mark.parametrize("backend", ["gluon", "triton"])
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
    preshuffle_weights,
    backend,
    device="cuda",
):
    if get_arch() != "gfx950" and get_arch() != "gfx1250":
        pytest.skip("Kernel not supported on this GPU.")
    if not is_fp4_avail():
        pytest.skip(f"FP4 kernels are not supported on {get_arch()}.")

    if backend == "gluon" and get_arch() != "gfx1250":
        pytest.skip(f"Gluon backend requires gfx1250, got {get_arch()}.")
    if hbm_swizzling:
        if get_arch() == "gfx950" and (n % 32 != 0 or k % (32 * 8) != 0):
            pytest.skip(
                f"Shape {m}x{n}x{k} is not supported for scale swizzling on gfx950."
            )
        # gfx1250 uses the n32k4 layout (scale kwidth 4), so K only needs
        # K//32 divisible by 4; gfx950 still needs 8.
        if get_arch() == "gfx1250" and (n % 32 != 0 or k % (32 * 4) != 0):
            pytest.skip(
                f"Shape {m}x{n}x{k} is not supported for scale swizzling on gfx1250."
            )

    if preshuffle_weights:
        if get_arch() != "gfx1250":
            pytest.skip("Preshuffling weights is only supported on gfx1250")
        if backend == "triton":
            pytest.skip("Preshuffled weights are decoded by the gluon kernel only")
        if n % 16 != 0 or (k // 2) % 32 != 0:
            pytest.skip(
                f"Preshuffling weights requires n divisible by 16 and k//2 divisible "
                f"by 32, got n={n}, k//2={k // 2}"
            )

    torch.manual_seed(0)

    weight_mxfp4 = "mxfloat4_e2m1"
    weight_dtype_str = weight_mxfp4[2:]

    weight_dtype = dtype_str_to_torch(weight_dtype_str)
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
        torch.bfloat16,
        torch.bfloat16,
        has_y_gammas,
        device=device,
    )
    x_ref, w_ref, bias_ref = x_tri.clone(), w_tri.clone(), bias_tri.clone()

    # downcast to mxfp4
    w_tri, w_scale_tri = downcast_to_mxfp(w_tri, weight_dtype, axis=1)
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)
    if hbm_swizzling:
        if get_arch() == "gfx1250":
            swizzle_mx_scale = "GFX1250_SCALE"
            w_scale_tri = preshuffle_moe_wscale(w_scale_tri)
        elif get_arch() == "gfx950":
            w_scale_tri, swizzle_mx_scale = shuffle_scale_moe(
                w_scale_tri,
                arch="gfx950",
                preshuffle_factor=32,
                scale_kwidth=8,
                return_layout=True,
            )
        else:
            assert False, "Unsupported architecture"
    else:
        swizzle_mx_scale = None
    if preshuffle_weights:
        w_tri = preshuffle_moe_weight(w_tri)

    x_tri, x_mx_scales_tri = mxfp4_quant(x_tri)
    x_ref = upcast_from_mxfp(x_tri, x_mx_scales_tri, torch.bfloat16, axis=-1)
    out_dtype = torch.bfloat16
    maxtol = None
    rmstol = None

    ref_y = moe_gemm_torch(
        x_ref, w_ref, bias_ref, rdata, gindx, sindx, gammas, apply_swiglu
    )

    # run kernel
    tri_y = moe_gemm_a4w4(
        x_tri,
        w_tri,
        x_mx_scales_tri,
        w_scale_tri,
        bias_tri,
        rdata,
        gindx,
        sindx,
        gammas,
        swizzle_mx_scale,
        preshuffle_weights,
        out_dtype,
        apply_swiglu,
        backend=backend,
    )
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)


# ---------------
# In-kernel gather correctness: in-kernel gather must match pre-gathered for all
# block sizes, including block_m=128 where a layout bug caused the gather index
# tensor to repeat only 32 unique values instead of covering all 128 rows, and
# block_m=64 where multi-block experts triggered a modular wrap-around in the
# clamp used to guard OOB GatherIndx reads.
# ---------------

_SKIP_GATHER = pytest.mark.skipif(
    get_arch() != "gfx1250",
    reason="In-kernel gather via gluon prefill kernel is gfx1250-only",
)


def _ep_routing(M, E, E_LOCAL, TOPK, block_m_cap, device):
    """Compute EP routing with a fixed random seed; returns (rd, gidx, sidx, gvalid, dst)."""
    g = torch.Generator(device=device).manual_seed(7)
    vals, ids = torch.topk(torch.randn(M, E, generator=g, device=device), TOPK, dim=-1)
    ids = ids.to(torch.int32).contiguous()
    wts = torch.softmax(vals, -1).contiguous()

    emap = torch.full((E + 1,), -1, dtype=torch.int32, device=device)
    emap[0:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=device)

    n_gates = M * TOPK
    block_m = max(16, min(triton.next_power_of_2(max(1, n_gates // E)), block_m_cap))
    bufs = _compute_expt_data_internal(E_LOCAL, n_gates, block_m, device)
    geo = EpScatterGeometry(
        src_token_map=torch.arange(M, dtype=torch.int32, device=device),
        max_tokens_per_rank=M,
        peer_rows=n_gates,
    )
    hist, gidx, sidx, gscal, gvalid, dst = ep_sort_routing(
        wts,
        ids,
        emap,
        E_LOCAL,
        torch.full((1,), M, dtype=torch.int32, device=device),
        M,
        TOPK,
        n_gates,
        bufs,
        ep_scatter_geometry=geo,
    )
    h = hist[:E_LOCAL]
    rd = RoutingData(block_m, gscal, h, E_LOCAL, TOPK, ExptData(h, bufs[0], bufs[1], bufs[2]))
    return rd, gidx, sidx, gvalid, dst


def _gemm1_gemm2_ikg_vs_pregathered(M, E, E_LOCAL, TOPK, D, I, block_m_cap, device):
    """Run the full GEMM1 → GEMM2 pipeline in both modes and compare the final output.

    Uses the preshuffled MXFP4 W4A4, SwiGLU, out_mx_quant (GEMM1) and
    preshuffled MXFP4 W4A4 (GEMM2) configuration, matching the shape used by
    DeepSeek-style models on gfx1250.  GEMM1 emits MXFP4-quantised intermediates
    (out_mx_quant=True), which GEMM2 consumes directly.  The final bfloat16
    output is what the test compares.
    """
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(M, D, generator=g, device=device, dtype=torch.bfloat16)
    xq, xs = mxfp4_quant(x)

    def _make_weight(n_out, n_in, seed):
        g2 = torch.Generator(device=device).manual_seed(seed)
        w = torch.randn(E_LOCAL, n_out, n_in, generator=g2, device=device, dtype=torch.bfloat16) / n_in**0.5
        q, s = mxfp4_quant(w.reshape(-1, n_in))
        q = q.view(E_LOCAL, n_out, n_in // 2)
        s = s.view(E_LOCAL, n_out, n_in // 32)
        wk = moe_weight_decode_view(moe_shuffle_weight(q.contiguous()).contiguous())
        sk = moe_shuffle_scale(s.contiguous()).transpose(-1, -2)
        return wk, sk

    w1k, s1k = _make_weight(2 * I, D, seed=1)
    w2k, s2k = _make_weight(D, I, seed=2)

    rd, gidx, sidx, gvalid, dst = _ep_routing(M, E, E_LOCAL, TOPK, block_m_cap, device)
    assert rd.block_m == block_m_cap, f"routing chose block_m={rd.block_m}, expected {block_m_cap}"

    gemm1_kw = dict(
        swizzle_mx_scale="GFX1250_SCALE",
        preshuffle_weights=True,
        out_mx_quant=True,
        apply_swiglu=True,
        alpha=1.0,
        limit=1e4,
        swiglu_add_residual=False,
    )
    gemm2_kw = dict(swizzle_mx_scale="GFX1250_SCALE", preshuffle_weights=True)

    def _run(in_kernel_gather):
        if in_kernel_gather:
            a, a_s, gi = xq, xs, gidx
        else:
            rows = gidx.to(torch.int64).clamp(0, M * TOPK - 1) // TOPK
            a, a_s, gi = xq[rows], xs[rows], None
        y1, y1s = moe_gemm_a4w4(a, w1k, a_s, s1k, None, rd, gather_indx=gi, **gemm1_kw)
        raw = torch.zeros(M * TOPK, D, dtype=torch.bfloat16, device=device)
        moe_gemm_a4w4(
            y1, w2k, y1s, s2k, None, rd,
            scatter_indx=sidx, gate_valid=gvalid,
            ep_scatter=EpCombineScatter(out=raw, dst_row=dst),
            **gemm2_kw,
        )
        return raw.view(M, TOPK, D)

    out_ikg = _run(in_kernel_gather=True)
    out_pg = _run(in_kernel_gather=False)

    cos = torch.nn.functional.cosine_similarity(
        out_ikg.float().flatten(), out_pg.float().flatten(), dim=0
    ).item()
    assert cos > 0.999, (
        f"in-kernel gather vs pre-gathered cosine={cos:.6f} (block_m={rd.block_m})"
    )


@_SKIP_GATHER
@pytest.mark.parametrize(
    "M, block_m_cap",
    [
        # block_m=32: baseline, one M-tile covers most experts fully
        (1024, 32),
        # block_m=64: multi-block experts; % M clamp was wrong for block_id >= 1
        (2048, 64),
        # block_m=128: gather layout only produced 32 unique indices (repeated 4×)
        (4096, 128),
        # block_m=128 with a larger token count
        (8192, 128),
    ],
)
def test_gluon_in_kernel_gather_matches_pregathered(M, block_m_cap, device="cuda"):
    """In-kernel gather must equal pre-gathered for all block sizes on gfx1250.

    Covers two distinct bugs fixed together:
    - block_m=128: GATHER_IDX_LAYOUT covered only 32 elements, so arange(128)
      repeated [0..31] four times, loading the same 32 activation rows for all
      128 rows of the tile.
    - block_m=64 (multi-block experts): gather indices were clamped with ``% M``
      instead of ``min(..., M-1)``, causing block_id >= 1 to wrap to wrong rows.
    """
    E, E_LOCAL, TOPK, D, I = 256, 128, 8, 2304, 2304
    _gemm1_gemm2_ikg_vs_pregathered(M, E, E_LOCAL, TOPK, D, I, block_m_cap, device)


@_SKIP_GATHER
def test_gluon_in_kernel_gather_multiblock_experts(device="cuda"):
    """Verify correct gather output for experts whose token count exceeds block_m.

    Uses block_m=64 and M=2048 so most experts have more than 64 tokens each
    (histogram: min~43, max~83), exercising block_id=1 for the majority of
    experts. The % M clamp bug caused block_id=1 to re-read rows 0..M_e-1
    again instead of rows 64..M_e-1.
    """
    E, E_LOCAL, TOPK, D, I = 256, 128, 8, 2304, 2304
    M, block_m_cap = 2048, 64
    _gemm1_gemm2_ikg_vs_pregathered(M, E, E_LOCAL, TOPK, D, I, block_m_cap, device)

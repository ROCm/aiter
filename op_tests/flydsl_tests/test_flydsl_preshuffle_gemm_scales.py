"""FP32 epilogue and per-128-K scales in flydsl_preshuffle_gemm_a8."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() not in ("gfx942", "gfx950"),
    reason="gfx942 or gfx950 required",
)

K = 2176
N = 2048
CONFIGS = [(1, 16, 64, 128, 1), (4, 32, 64, 128, 6)]


def _make(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(1)
    a = torch.randn(m, k, generator=g, device="cuda") * 0.5
    b = torch.randn(n, k, generator=g, device="cuda") * 0.5
    aq, xs = aiter.per_tensor_quant(a, quant_dtype=dtypes.fp8)
    bq, ws = aiter.per_tensor_quant(b, quant_dtype=dtypes.fp8)
    ref = (aq.float() * xs.float()) @ (bq.float() * ws.float()).T
    return aq, xs.float(), shuffle_weight(bq, layout=(16, 16)), ws.float(), ref


def _shape(s, rows):
    return {
        "expanded": lambda: s.reshape(1, 1).expand(rows, 1).contiguous(),
        "scalar1": lambda: s.reshape(1),
        "scalar11": lambda: s.reshape(1, 1),
    }


@pytest.mark.parametrize("form", ["expanded", "scalar1", "scalar11"])
@pytest.mark.parametrize("m,tm,tn,tk,sk", CONFIGS)
def test_per_tensor_scale_forms(m, tm, tn, tk, sk, form):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    aq, xs, wq, ws, ref = _make(m, N, K)
    out = torch.full((m, N), float("nan"), device="cuda", dtype=dtypes.bf16)
    flydsl_preshuffle_gemm_a8(
        aq,
        wq,
        _shape(xs, m)[form](),
        _shape(ws, N)[form](),
        out,
        tm,
        tn,
        tk,
        0,
        0,
        0,
        lds_stage=2,
        enable_scheduler=True,
        split_k=sk,
    )
    torch.cuda.synchronize()
    o = out.float()
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    assert torch.isfinite(o).all()
    assert (o - ref).abs().le(tol).all()


def test_wrong_length_scale_raises():
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    m = 4
    aq, _xs, wq, ws, _ = _make(m, N, K)
    out = torch.empty((m, N), device="cuda", dtype=dtypes.bf16)
    bad = torch.ones(m + 1, 1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="x_scale"):
        flydsl_preshuffle_gemm_a8(aq, wq, bad, ws.reshape(1), out, 32, 64, 128, 0, 0, 0)


BLOCKSCALE_CONFIGS = [
    (m, 2048, k, 32, 128, 1)
    for m in (1, 4, 16)
    for k in (2176, 7168)
] + [
    (1, 2048, 2176, 32, 128, 6),
    (4, 2048, 7168, 32, 256, 1),
    (1, 2048, 2176, 16, 128, 1),
    (16, 4096, 7168, 32, 128, 1),
]


@pytest.mark.parametrize("m,n,k,tm,tk,sk", BLOCKSCALE_CONFIGS)
def test_blockscale(m, n, k, tm, tk, sk):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    g = torch.Generator(device="cuda").manual_seed(123)
    aq = (torch.rand((m, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    bq = (torch.rand((n, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    # Distinct row/column and K-block scales expose transposition and pairing errors.
    kb = torch.arange(k // 128, device="cuda", dtype=torch.float32)
    rows = torch.arange(m, device="cuda", dtype=torch.float32)
    cols = torch.arange(n // 128, device="cuda", dtype=torch.float32)
    xs = (0.5 + (kb[:, None] % 7) * 0.125 + rows[None, :] * 0.0625).contiguous()
    ws = (0.75 + (cols[:, None] % 5) * 0.125 + (kb[None, :] % 11) * 0.0625).contiguous()
    a = aq.float() * xs.T.repeat_interleave(128, dim=1)
    b = bq.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    ref = a @ b.T
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtypes.bf16)
    flydsl_preshuffle_gemm_a8(
        aq,
        shuffle_weight(bq, layout=(16, 16)),
        xs,
        ws,
        out,
        tm,
        64,
        tk,
        0,
        0,
        0,
        lds_stage=2,
        enable_scheduler=True,
        split_k=sk,
        scale_mode="blockscale",
    )
    torch.cuda.synchronize()
    o = out.float()
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    nonfinite = (~torch.isfinite(o)).sum().item()
    bad = ((o - ref).abs() > tol).sum().item()
    assert nonfinite == 0, f"nonfinite={nonfinite}"
    assert bad == 0, f"bad={bad}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx942",
    reason="gfx942 async-copy rejection",
)
def test_gfx942_async_copy_raises():
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    aq, xs, wq, ws, _ = _make(1, N, K)
    out = torch.empty((1, N), device="cuda", dtype=dtypes.bf16)
    with pytest.raises(ValueError, match="LDS-direct loads unavailable on gfx942"):
        flydsl_preshuffle_gemm_a8(
            aq, wq, xs, ws, out, 32, 64, 128, use_async_copy=1
        )


@pytest.mark.parametrize("m,k,sk", [(1, 2176, 6), (4, 7168, 1), (16, 2176, 1)])
def test_per_row_scales(m, k, sk):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    g = torch.Generator(device="cuda").manual_seed(1)
    aq = (torch.rand((m, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    bq = (torch.rand((N, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    wq = shuffle_weight(bq, layout=(16, 16))
    xs = torch.linspace(0.5, 1.5, m, device="cuda").reshape(m, 1)
    ws = torch.linspace(0.75, 1.25, N, device="cuda").reshape(N, 1)
    ref = (aq.float() * xs) @ (bq.float() * ws).T
    out = torch.full((m, N), float("nan"), device="cuda", dtype=dtypes.bf16)
    flydsl_preshuffle_gemm_a8(
        aq, wq, xs, ws, out, 32, 64, 128, 0, 0, 0, split_k=sk
    )
    torch.cuda.synchronize()
    o = out.float()
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    assert torch.isfinite(o).all()
    assert (o - ref).abs().le(tol).all()

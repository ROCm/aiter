"""Per-tensor / per-row scale normalization in flydsl_preshuffle_gemm_a8."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="gfx950 required",
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

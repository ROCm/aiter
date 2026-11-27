# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The A6W6 GEMM as a ``GemmHooks`` host (``gemm_a6w6_4wave_host``).

The plain projection is checked against the standalone A6W6 GEMM bit for bit. The bias,
Q/K prep and maxima epilogues are checked against an FP32 PyTorch reference of the
dequantized MXFP6 operands, and the persistent grid against one block per workgroup.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.gemm_op_a6w6 import _ceil, dequant_mxfp6_torch, quant_mxfp6_torch

HEAD = 128
EPS = 1e-6


def _is_gfx950():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="the A6W6 host needs gfx950")


def _case(m, n, k, bias, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    weight = 1 + 0.1 * torch.randn(n, device="cuda", generator=gen)
    angles = torch.rand((m, 64), device="cuda", generator=gen) * 6.28
    return {
        "x": x,
        "w": w,
        "bias": torch.randn(n, device="cuda", generator=gen) * 0.1 if bias else None,
        "params": {"weight": weight},
        "tables": {"cos": angles.cos(), "sin": angles.sin()},
    }


def _dequantized(t):
    """The FP32 values the MFMA multiplies: Hadamard-rotated, MXFP6-rounded, K padded.
    The rotation cancels in the product."""
    t = F.pad(t, (0, _ceil(t.shape[1], 128) - t.shape[1]))
    return dequant_mxfp6_torch(*quant_mxfp6_torch(t))


def _reference(case, spec=None, bias=False):
    """Stored projection, the row-factor op's per-head partials and per-head maxima of the
    stored output."""
    acc = F.linear(_dequantized(case["x"]), _dequantized(case["w"]))
    if (spec.bias if spec is not None else bias) and case["bias"] is not None:
        acc = acc + case["bias"]
    m, n = acc.shape
    y = acc.view(m, n // HEAD, HEAD)
    partials = None
    if spec is not None:
        y, partials = spec.reference(y, case["params"], case["tables"])
    stored = y.reshape(m, n).to(torch.bfloat16)
    maxima = stored.float().abs().view(m, n // HEAD, HEAD).amax(-1)
    return stored, partials, maxima


def _launch(case, spec=None, *, bias=False, amax=False, num_cu=0):
    """The host GEMM into NaN-filled outputs, so unwritten elements fail."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a6w6_4wave_host import (
        compile_a6w6_host_4w,
        host_weight_order,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    x, w = case["x"], case["w"]
    (m, k), n = x.shape, w.shape[0]
    a, a_scale = aiter.quant_mxfp6_gemm(x)
    b, b_scale = aiter.quant_mxfp6_gemm(w[host_weight_order(n).to(w.device)])
    tables = None
    # The host applies no column scales, so its column operand starts with the bias; a
    # plain projection reads none of it.
    if spec is not None:
        columns = spec.pack_columns(
            None, case["bias"], [case["params"]], n // HEAD, n=n
        )
        tables = spec.pack_tables(case["tables"])
    else:
        columns = case["bias"] if bias else torch.zeros(1, device="cuda")
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    partials = torch.full((2, m, n // HEAD), float("nan"), device="cuda")
    kernel = compile_a6w6_host_4w(
        K=_ceil(k, 128), num_cu=num_cu, prep=spec, bias=bias, amax=amax
    )
    _run_compiled(
        kernel,
        a.view(-1),
        b.view(-1),
        out.view(-1),
        a_scale.view(-1),
        b_scale.view(-1),
        columns,
        # Without tables the kernel never reads its tables argument.
        (tables if tables is not None else columns).view(-1),
        partials.view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    return out, partials


def _close(expected, actual, rtol, label, outliers=0.0):
    """``actual`` within ``rtol`` of ``expected``, but for a fraction ``outliers``, and
    within 1e-3 RMS. The kernel and PyTorch add the K products in different orders: an
    FP32 result near a BF16 rounding boundary, or small after the Hadamard's sums,
    lands a step or two away."""
    e, a = expected.float(), actual.float()
    bad = ~torch.isclose(a, e, rtol=rtol, atol=1e-6)
    assert (
        bad.float().mean().item() <= outliers
    ), f"{label}: {int(bad.sum())} of {bad.numel()} outside rtol {rtol}"
    rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    assert rms <= 1e-3, f"{label}: RMS error {rms:.3g}"


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256, 128),
        (300, 512, 1000),
        (1024, 768, 1280),
        (9472, 15360, 5120),
        # N % 256 == 128: the last block's second half is past N.
        (256, 128, 256),
        (300, 384, 1000),
    ],
)
@pytest.mark.parametrize("persistent", [False, True])
@torch.no_grad()
def test_plain_matches_standalone_a6w6(shape, persistent):
    m, n, k = shape
    case = _case(m, n, k, False, seed=m + n + k)
    a, a_scale = aiter.quant_mxfp6_gemm(case["x"])
    b, b_scale = aiter.quant_mxfp6_gemm(case["w"])
    expected = aiter.gemm_a6w6(a, b, a_scale, b_scale, m, n, k)
    out, _ = _launch(case, num_cu=get_cu_num() if persistent else 0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


PREP_CASES = [
    (257, 512, 512, norm, rope, hadamard, True, False)
    for norm, rope, hadamard in itertools.product(
        ("none", "rms_per_head", "rms_across_heads"),
        ("none", "interleaved", "rotate_half"),
        (True, False),
    )
] + [
    # Without a bias, then persistent at Wan's 40-head and FLUX.3's 24-head projections.
    (257, 512, 512, "rms_across_heads", "interleaved", True, False, False),
    (257, 512, 512, "rms_per_head", "rotate_half", True, False, False),
    (257, 512, 512, "rms_per_head", "interleaved", True, False, False),
    (1184, 5120, 5120, "rms_across_heads", "interleaved", True, True, True),
    (1184, 5120, 5120, "rms_per_head", "rotate_half", True, True, True),
    (576, 3072, 3072, "rms_per_head", "interleaved", True, False, True),
    # One and three heads: N % 256 == 128.
    (257, 128, 512, "rms_per_head", "interleaved", True, True, True),
    (257, 384, 512, "rms_across_heads", "rotate_half", False, True, False),
]


@pytest.mark.parametrize("m,n,k,norm,rope,hadamard,bias,persistent", PREP_CASES)
@torch.no_grad()
def test_prep_matches_pytorch(m, n, k, norm, rope, hadamard, bias, persistent):
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec

    spec = PrepSpec(norm, rope, bias, hadamard, eps=EPS)
    case = _case(m, n, k, bias, seed=m + n + k)
    stored, row_partials, maxima = _reference(case, spec)
    out, partials = _launch(case, spec, num_cu=get_cu_num() if persistent else 0)
    _close(stored, out, 2**-7, f"{spec}: output", outliers=1e-3)
    _close(maxima, partials[1], 2**-7, f"{spec}: head maxima", outliers=1e-3)
    if spec.row_factor is not None:
        _close(row_partials, partials[0], 1e-3, f"{spec}: row partials", outliers=1e-3)


@pytest.mark.parametrize(
    "n,bias,amax",
    [
        (512, True, False),
        (512, False, True),
        (512, True, True),
        (128, True, True),
        (384, False, True),
    ],
)
@torch.no_grad()
def test_bias_and_maxima_match_pytorch(n, bias, amax):
    m, k = 300, 1000
    case = _case(m, n, k, bias, seed=11)
    stored, _, maxima = _reference(case, bias=bias)
    out, partials = _launch(case, bias=bias, amax=amax)
    _close(stored, out, 2**-7, "output", outliers=1e-3)
    if amax:
        _close(maxima, partials[1], 2**-7, "head maxima", outliers=1e-3)


@torch.no_grad()
def test_roles_under_a_skewed_block_order():
    """Two roles at M % 256 != 0 on 16 workgroups. The tile hook holds workgroups 0-3 in
    role 0 until 12 blocks of role 1 are counted, so they store role 0's last blocks after
    role 1's first: a row of role 0 past M must not land in role 1's plane."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr

    from aiter.ops.flydsl.kernels.communication_ops_utils import (
        spin_until_ge_i32_agent,
    )
    from aiter.ops.flydsl.kernels.gemm_a6w6_4wave_host import (
        compile_a6w6_host_4w,
        host_weight_order,
    )
    from aiter.ops.flydsl.kernels.gemm_hooks import GemmHooks
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    m, k, roles, role_cols, num_cu = 257, 512, 2, 2560, 16
    n = roles * role_cols
    held, released = 4, 12

    @flyc.jit
    def hold(role, aux, args, c_m, row_base, col_base):
        # A trace-time branch around a runtime one; they do not merge.
        if const_expr(role == 0):  # noqa: SIM102
            if fx.Int32(fx.block_idx.x) < held:
                if fx.thread_idx.x == 0:
                    spin_until_ge_i32_agent(aux + 4, released)
                fx.barrier()

    spec = PrepSpec("rms_across_heads", "interleaved", True, False, eps=EPS)
    case = _case(m, n, k, True, seed=17)
    params = {"weight": case["params"]["weight"][:role_cols]}
    columns = spec.pack_columns(None, case["bias"], [params], role_cols // HEAD, n=n)
    tables = spec.pack_tables(case["tables"])
    a, a_scale = aiter.quant_mxfp6_gemm(case["x"])
    b, b_scale = aiter.quant_mxfp6_gemm(case["w"][host_weight_order(n).to("cuda")])
    out = torch.full(
        (roles, m, role_cols), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    partials = torch.full((2, m, n // HEAD), float("nan"), device="cuda")
    # aux: the block counters, then from byte 64 the last role's per-column maxima.
    row_blocks = -(-m // 128)
    aux = torch.full((16 + row_blocks * role_cols,), float("nan"), device="cuda")
    aux.view(torch.int32)[:16] = 0
    kernel = compile_a6w6_host_4w(
        K=_ceil(k, 128),
        num_cu=num_cu,
        prep=spec,
        roles=roles,
        role_cols=role_cols,
        hooks=GemmHooks(tile=hold, key="hold"),
        col_amax_offset=64,
    )
    _run_compiled(
        kernel,
        a.view(-1),
        b.view(-1),
        out.view(-1),
        a_scale.view(-1),
        b_scale.view(-1),
        columns,
        tables.view(-1),
        partials.view(-1),
        aux.data_ptr(),
        *([0] * 6),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()

    acc = F.linear(_dequantized(case["x"]), _dequantized(case["w"])) + case["bias"]
    heads = role_cols // HEAD
    q = acc[:, :role_cols].reshape(m, heads, HEAD)
    q, sums = spec.reference(q, params, case["tables"])
    stored_q = q.reshape(m, role_cols).to(torch.bfloat16)
    stored_v = acc[:, role_cols:].to(torch.bfloat16)
    _close(stored_q, out[0], 2**-7, "role 0", outliers=1e-3)
    _close(stored_v, out[1], 2**-7, "role 1", outliers=1e-3)
    _close(sums, partials[0, :, :heads], 1e-3, "role 0 row partials", outliers=1e-3)
    for role, stored in enumerate((out[0], out[1])):
        maxima = stored.float().abs().view(m, heads, HEAD).amax(-1)
        assert torch.equal(partials[1, :, role * heads : (role + 1) * heads], maxima)
    blocks = -(-m // 256) * (role_cols // 256)
    assert aux.view(torch.int32)[:roles].tolist() == [blocks] * roles
    padded = F.pad(out[1].float().abs(), (0, 0, 0, row_blocks * 128 - m))
    expected = padded.view(row_blocks, 128, role_cols).amax(1)
    assert torch.equal(aux[16:].view(row_blocks, role_cols), expected)


@torch.no_grad()
def test_persistent_matches_one_block_per_workgroup():
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec

    spec = PrepSpec("rms_across_heads", "interleaved", True, True, eps=EPS)
    case = _case(1184, 5120, 5120, True, seed=3)
    one = _launch(case, spec)
    persistent = _launch(case, spec, num_cu=get_cu_num())
    for a, b in zip(one, persistent):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@torch.no_grad()
def test_each_device_runs_its_own_launcher():
    """cuda:0 runs the host and the standalone GEMM first; cuda:1 then matches them bit
    for bit, the standalone GEMM with cuda:0 still current."""
    m, n, k = 300, 384, 1000
    case = _case(m, n, k, False, seed=7)
    a, a_scale = aiter.quant_mxfp6_gemm(case["x"])
    b, b_scale = aiter.quant_mxfp6_gemm(case["w"])
    operands = (a, b, a_scale, b_scale)
    expected = (_launch(case)[0], aiter.gemm_a6w6(*operands, m, n, k))
    with torch.cuda.device(1):
        other = {"x": case["x"].to("cuda:1"), "w": case["w"].to("cuda:1"), "bias": None}
        host = _launch(other)[0]
    standalone = aiter.gemm_a6w6(*(t.to("cuda:1") for t in operands), m, n, k)
    for want, got in zip(expected, (host, standalone)):
        assert torch.equal(want.cpu().view(torch.int16), got.cpu().view(torch.int16))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

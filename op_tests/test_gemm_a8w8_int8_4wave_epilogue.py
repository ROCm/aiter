# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The epilogues of the 4-wave INT8 GEMM (``compile_int8_gemm_4w``): the Q/K prep, the
per-head maxima and the multi-role kernel that hosts ``GemmHooks``.

Each is checked against an FP32 PyTorch reference of the INT8 product: the prep against
``PrepSpec.reference``, the multi-role kernel under a tile hook that skews its block
order. The model-size prep rows are also timed.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from aiter import logger
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest

HEAD = 128
EPS = 1e-6


def _is_gfx950():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="the 4-wave INT8 GEMM needs gfx950"
)


def _case(m, n, k, bias, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(-31, 32, (m, k), dtype=torch.int8, device="cuda", generator=gen)
    w = torch.randint(-31, 32, (n, k), dtype=torch.int8, device="cuda", generator=gen)
    angles = torch.rand((m, 64), device="cuda", generator=gen) * 6.28
    return {
        "x": x,
        "w": w,
        "w_shuffled": shuffle_weight(w, layout=(16, 16)).contiguous(),
        "x_scale": torch.rand((m, 1), device="cuda", generator=gen) * 1e-3 + 1e-4,
        "w_scale": torch.rand((1, n), device="cuda", generator=gen) * 1e-3 + 1e-4,
        "bias": torch.randn(n, device="cuda", generator=gen) * 0.1 if bias else None,
        "params": {"weight": 1 + 0.1 * torch.randn(n, device="cuda", generator=gen)},
        "tables": {"cos": angles.cos(), "sin": angles.sin()},
    }


def _projection(case):
    """The scaled INT8 product with its bias, in FP32; the INT8 sums are exact."""
    acc = F.linear(case["x"].float(), case["w"].float())
    acc = acc * (case["x_scale"] * case["w_scale"])
    return acc if case["bias"] is None else acc + case["bias"]


def _reference(acc, spec=None, params=None, tables=None):
    """The stored projection, the row-factor op's per-head partials and the per-head
    maxima of the stored output."""
    m, n = acc.shape
    y = acc.reshape(m, n // HEAD, HEAD)
    partials = None
    if spec is not None:
        y, partials = spec.reference(y, params, tables)
    stored = y.reshape(m, n).to(torch.bfloat16)
    maxima = stored.float().abs().view(m, n // HEAD, HEAD).amax(-1)
    return stored, partials, maxima


def _check(expected, actual, rtol, label, atol=1e-6):
    """Every element within ``rtol``, so an unwritten (NaN) one fails. PyTorch rounds
    the FP32 math in another order, so a BF16 value may land one step (2 ** -7) away."""
    bad = ~torch.isclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    assert not bad.any(), f"{label}: {int(bad.sum())} of {bad.numel()} outside rtol"


def _launch(case, spec=None, *, amax=False, num_cu=0):
    """A launch of the GEMM into NaN-filled outputs, and the outputs."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    (m, k), n = case["x"].shape, case["w"].shape[0]
    bias = case["bias"] is not None
    if spec is not None:
        columns = spec.pack_columns(
            case["w_scale"], case["bias"], [case["params"]], n // HEAD
        )
        tables = spec.pack_tables(case["tables"])
    else:
        columns = torch.cat([case["w_scale"].reshape(-1)] + ([case["bias"]] * bias))
        tables = None
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    partials = torch.full((2, m, n // HEAD), float("nan"), device="cuda")
    kernel = compile_int8_gemm_4w(K=k, num_cu=num_cu, bias=bias, prep=spec, amax=amax)
    args = (
        case["x"].view(-1),
        case["w_shuffled"].view(-1),
        out.view(-1),
        case["x_scale"].view(-1),
        columns,
        # Without tables the kernel never reads its tables argument.
        (tables if tables is not None else columns).view(-1),
        partials.view(-1),
        m,
        n,
    )

    def launch():
        _run_compiled(kernel, *args, fx.Stream(torch.cuda.current_stream()))
        return out, partials

    return launch


def _check_prep(spec, case, launch):
    stored, row_partials, maxima = _reference(
        _projection(case), spec, case["params"], case["tables"]
    )
    out, partials = launch()
    _check(stored, out, 2**-7, f"{spec}: output")
    _check(maxima, partials[1], 2**-7, f"{spec}: head maxima")
    if spec.row_factor is not None:
        _check(row_partials, partials[0], 1e-5, f"{spec}: row partials", atol=0)


PREP_CASES = [
    (257, 512, 512, norm, rope, hadamard, True, False)
    for norm, rope, hadamard in itertools.product(
        ("none", "rms_per_head", "rms_across_heads"),
        ("none", "interleaved", "rotate_half"),
        (True, False),
    )
] + [
    # Without a bias.
    (257, 512, 512, "rms_across_heads", "interleaved", True, False, False),
    (257, 512, 512, "rms_per_head", "rotate_half", True, False, False),
    (257, 512, 512, "rms_per_head", "interleaved", True, False, False),
    # An odd head count: the last block's second tile is past N.
    (257, 384, 512, "rms_across_heads", "none", False, False, False),
    (257, 384, 512, "rms_across_heads", "rotate_half", True, True, True),
]


@pytest.mark.parametrize("m,n,k,norm,rope,hadamard,bias,persistent", PREP_CASES)
@torch.no_grad()
def test_prep_matches_pytorch(m, n, k, norm, rope, hadamard, bias, persistent):
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec

    spec = PrepSpec(norm, rope, bias, hadamard, eps=EPS)
    case = _case(m, n, k, bias, seed=m + n + k)
    num_cu = get_cu_num() if persistent else 0
    _check_prep(spec, case, _launch(case, spec, num_cu=num_cu))


@torch.no_grad()
def test_prep_with_a_non_default_eps():
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec

    spec = PrepSpec("rms_across_heads", "interleaved", True, True, eps=1e-5)
    case = _case(257, 512, 512, True, seed=5)
    _check_prep(spec, case, _launch(case, spec))


# Wan's 40-head and FLUX.3's 24-head projections, persistent (FLUX.3: per-head norm,
# interleaved RoPE, no bias).
MODEL_ROWS = [
    (1184, 5120, 5120, "rms_across_heads", "interleaved", True),
    (1184, 5120, 5120, "rms_per_head", "rotate_half", True),
    (576, 3072, 3072, "rms_per_head", "interleaved", False),
]


@pytest.mark.parametrize("m,n,k,norm,rope,bias", MODEL_ROWS)
@torch.no_grad()
def test_prep_model_rows(m, n, k, norm, rope, bias):
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec

    spec = PrepSpec(norm, rope, bias, True, eps=EPS)
    case = _case(m, n, k, bias, seed=m + n + k)
    launch = _launch(case, spec, num_cu=get_cu_num())
    _, us = run_perftest(launch, num_iters=50)
    _check_prep(spec, case, launch)
    logger.info(f"{spec} at {m}x{n}x{k}, persistent: {us:.1f} us")


@pytest.mark.parametrize("m,n,k,bias", [(257, 384, 512, True), (300, 512, 1024, False)])
@torch.no_grad()
def test_amax_matches_pytorch(m, n, k, bias):
    case = _case(m, n, k, bias, seed=m + n)
    stored, _, maxima = _reference(_projection(case))
    out, partials = _launch(case, amax=True)()
    _check(stored, out, 2**-7, "output")
    _check(maxima, partials[1], 2**-7, "head maxima")
    assert partials[0].isnan().all(), "amax writes only the maxima"


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
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.gemm_hooks import GemmHooks
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    m, k, roles, role_cols, num_cu = 257, 512, 2, 2560, 16
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
    case = _case(m, roles * role_cols, k, True, seed=17)
    params = {"weight": case["params"]["weight"][:role_cols]}
    columns = spec.pack_columns(
        case["w_scale"], case["bias"], [params], role_cols // HEAD
    )
    tables = spec.pack_tables(case["tables"])
    out = torch.full(
        (roles, m, role_cols), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    partials = torch.full(
        (2, m, roles * role_cols // HEAD), float("nan"), device="cuda"
    )
    # aux: the block counters, then from byte 64 the last role's per-column maxima.
    row_blocks = -(-m // 128)
    aux = torch.full((16 + row_blocks * role_cols,), float("nan"), device="cuda")
    aux.view(torch.int32)[:16] = 0
    kernel = compile_int8_gemm_4w(
        K=k,
        num_cu=num_cu,
        prep=spec,
        roles=roles,
        role_cols=role_cols,
        hooks=GemmHooks(tile=hold, key="hold"),
        col_amax_offset=64,
    )
    _run_compiled(
        kernel,
        case["x"].view(-1),
        case["w_shuffled"].view(-1),
        out.view(-1),
        case["x_scale"].view(-1),
        columns,
        tables.view(-1),
        partials.view(-1),
        aux.data_ptr(),
        *([0] * 6),
        m,
        roles * role_cols,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()

    acc = _projection(case)
    stored_q, sums, maxima_q = _reference(
        acc[:, :role_cols], spec, params, case["tables"]
    )
    stored_v, _, maxima_v = _reference(acc[:, role_cols:])
    heads = role_cols // HEAD
    _check(stored_q, out[0], 2**-7, "role 0")
    _check(stored_v, out[1], 2**-7, "role 1")
    _check(sums, partials[0, :, :heads], 1e-5, "role 0 row partials", atol=0)
    _check(maxima_q, partials[1, :, :heads], 2**-7, "role 0 head maxima")
    _check(maxima_v, partials[1, :, heads:], 2**-7, "role 1 head maxima")
    blocks = -(-m // 256) * (role_cols // 256)
    assert aux.view(torch.int32)[:roles].tolist() == [blocks] * roles
    padded = F.pad(out[1].float().abs(), (0, 0, 0, row_blocks * 128 - m))
    expected = padded.view(row_blocks, 128, role_cols).amax(1)
    assert torch.equal(aux[16:].view(row_blocks, role_cols), expected)


def test_rejects_what_it_would_run_otherwise():
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.gemm_hooks import GemmHooks
    from aiter.ops.flydsl.kernels.qk_prep import PrepSpec

    spec = PrepSpec("rms_across_heads", "none", False, False)
    hooks = GemmHooks(role_end=lambda role, aux, args, c_m: None, key="noop")
    with pytest.raises(ValueError, match="hooks need roles > 1"):
        compile_int8_gemm_4w(K=512, hooks=hooks)
    with pytest.raises(ValueError, match="col_amax_offset needs roles > 1"):
        compile_int8_gemm_4w(K=512, col_amax_offset=0)
    with pytest.raises(ValueError, match="amax is for an output without a prep"):
        compile_int8_gemm_4w(K=512, prep=spec, amax=True)
    with pytest.raises(ValueError, match="unsupported on this device"):
        compile_int8_gemm_4w(
            K=512,
            num_cu=2 * get_cu_num(),
            prep=spec,
            roles=2,
            role_cols=256,
            hooks=hooks,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@torch.no_grad()
def test_each_device_runs_its_own_launcher():
    """cuda:0 runs the kernel and the op first; cuda:1 then matches them bit for bit, the
    op with cuda:0 still current."""
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave import (
        default_4wave_kernel_name,
        run_gemm_a8w8_bpreshuffle_4wave,
    )

    m, n, k = 300, 512, 512
    case = _case(m, n, k, True, seed=5)
    keys = ("x", "w", "w_shuffled", "x_scale", "w_scale", "bias")
    other = {key: case[key].to("cuda:1") for key in keys}

    def op(c):
        out = torch.empty((m, n), dtype=torch.bfloat16, device=c["x"].device)
        name = default_4wave_kernel_name(m, n, k)
        return run_gemm_a8w8_bpreshuffle_4wave(
            c["x"], c["w_shuffled"], c["x_scale"], c["w_scale"], out, name, c["bias"]
        )

    expected = (_launch(case, amax=True)()[0], op(case))
    with torch.cuda.device(1):
        kernel = _launch(other, amax=True)()[0]
    for want, got in zip(expected, (kernel, op(other))):
        assert torch.equal(want.cpu().view(torch.int16), got.cpu().view(torch.int16))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

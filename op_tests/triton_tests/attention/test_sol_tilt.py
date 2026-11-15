# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Sol-Attn Jensen cap and low-rank pooled value tilt: the public wrappers of sol_tilt.py.

Each wrapper is checked against a Torch reference of its docstring. The tilt planes are checked
through their documented decoding: for every full block, sum_i u2 u3 q_multiplier B_i A_i^T, with
A and B decoded as mean_k and mean_v are, must be the rank-r truncation of softmax_scale * Cov(v, k)
that the tilt stands in for. The packed formats cannot be decoded here, so their planes are checked
for layout and for the zeroed scale backing that the kernels read past the last row. Every wrapper
must also trace under torch.compile(fullgraph=True) and give the eager result.

The ASM kernels are not involved. The packed formats need the gfx950 packers.
"""

import contextlib
import math

import pytest
import torch
import torch._dynamo

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4_quant import rotate_activation_hd128
from aiter.ops.triton.attention import sol_tilt
from aiter.ops.triton.attention.sol_tilt import (
    sol_jensen_cap,
    sol_q_abs_max,
    sol_tilt_factors,
    sol_tilt_planes,
    sol_tilt_scaled_params,
    sol_tilt_valu_params,
)
from aiter.ops.triton.attention.utils import (
    SOL_ATTN_FP8_VAR_DIVISOR,
    SOL_ATTN_TS_KV,
    SOL_ATTN_TS_QO,
    sol_prepare,
)
from aiter.ops.triton.quant import mxfp6_fmha_pack as fp6

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
HEAD_DIM = 128
BETA = 0.5
PACKED_K = ("mxfp4", "mxfp6")
PACKED_V = ("mxfp4_fp6_p", "mxfp6_fp6_p")

requires_gfx950 = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="the MXFP4 and MXFP6 packers are gfx950 kernels"
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _allocated_tail(x):
    """The bytes of x's allocation past the last element x addresses, as uint8.

    The packed views are strided over their backing (head-major, with tile padding between the
    heads), so their end is their extent, not their element count.
    """
    extent = (
        1 + sum((n - 1) * s for n, s in zip(x.shape, x.stride())) if x.numel() else 0
    )
    return _bytes(x)[extent * x.element_size() :]


@contextlib.contextmanager
def _poisoned_allocations():
    """Fill what torch.empty, empty_like and new_empty return with 0xFF, so an unwritten byte
    reads nonzero and a zero byte is one its producer wrote."""
    originals = torch.empty, torch.empty_like, torch.Tensor.new_empty

    def poisoned(make):
        def make_poisoned(*args, **kwargs):
            out = make(*args, **kwargs)
            if out.is_cuda and out.numel():
                _bytes(out).fill_(0xFF)
            return out

        return make_poisoned

    torch.empty, torch.empty_like, torch.Tensor.new_empty = map(poisoned, originals)
    try:
        yield
    finally:
        torch.empty, torch.empty_like, torch.Tensor.new_empty = originals


def _bytes(x):
    """x's allocation from x's first element on, as uint8."""
    storage = torch.empty(0, dtype=torch.uint8, device=x.device)
    storage.set_(x.untyped_storage(), 0, (x.untyped_storage().nbytes(),))
    return storage[x.storage_offset() * x.element_size() :]


def _plane_bytes(x):
    """The bytes one plane of x's layout spans: its batches at x's batch stride."""
    return x.shape[0] * x.stride(0) * x.element_size()


def _plane(view, stride_bytes, index):
    """Plane `index` of a (1 + rank)-plane allocation whose plane 0 is `view`."""
    step, rest = divmod(index * stride_bytes, view.element_size())
    assert rest == 0
    return view.as_strided(view.shape, view.stride(), view.storage_offset() + step)


def _planted_kv(batch, seqlen_k, heads, rank, seed=0):
    """BF16 K and V in real units whose block covariance has a clear rank-`rank` part.

    Each token draws `rank` shared factors that drive both K and V along fixed per-head
    directions, with falling strengths, so the leading singular values stand clear of the noise.
    K and V also get a per-head offset, so the factors are taken from centered blocks.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float32, device="cuda")

    strength = 4.0 / torch.arange(1, rank + 1, dtype=torch.float32, device="cuda")
    z = randn(batch, seqlen_k, heads, rank) * strength
    k_dirs = randn(heads, rank, HEAD_DIM)
    v_dirs = randn(heads, rank, HEAD_DIM)
    k = torch.einsum("bshr,hrd->bshd", z, k_dirs) + 0.3 * randn(
        batch, seqlen_k, heads, HEAD_DIM
    )
    v = torch.einsum("bshr,hrd->bshd", z, v_dirs) + 0.3 * randn(
        batch, seqlen_k, heads, HEAD_DIM
    )
    k = k + randn(1, 1, heads, HEAD_DIM)
    v = v + randn(1, 1, heads, HEAD_DIM)
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def _per_tensor(x, fmt):
    """(codes, descale) of a per-tensor format, or (x, 1.0) for BF16."""
    if fmt == "bf16":
        return x, 1.0
    limit = 127.0 if fmt == "int8" else FP8_MAX
    descale = x.float().abs().amax().item() / limit
    codes = x.float() / descale
    if fmt == "int8":
        return codes.round().clamp(-127, 127).to(torch.int8), descale
    return codes.to(FP8), descale


def _mx(x):
    """(MXFP8 codes, E8M0 1x32 scale, dequantized BF16) of x."""
    groups = x.float().unflatten(-1, (-1, 32))
    exponent = torch.ceil(
        torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(2.0**-100) / FP8_MAX)
    )
    codes = (groups / torch.exp2(exponent)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    scale = (exponent.squeeze(-1) + 127).to(torch.uint8)
    dequantized = codes.float() * torch.exp2(exponent)
    return codes.flatten(-2), scale, dequantized.flatten(-2).to(torch.bfloat16)


def make_tilt_inputs(
    k_format,
    v_format,
    *,
    batch=1,
    seqlen_q=512,
    seqlen_k=4 * SOL_ATTN_TS_KV + 37,
    heads=2,
    kv_tile=SOL_ATTN_TS_KV,
    rank=2,
    softmax_scale=0.125,
    q_multiplier=1.0,
    seed=0,
):
    """The keyword arguments of sol_tilt_planes for one recipe, from sol_prepare's pooled K/V.

    K and V are planted as in _planted_kv and encoded in k_format and v_format. The factors are
    taken in the basis the kernel holds K and V in: the stored codes of a per-tensor format, the
    dequantized MXFP8 K, and the BF16 source of a packed format. Q is BF16 in that basis.
    """
    k_real, v_real = _planted_kv(batch, seqlen_k, heads, rank, seed)
    q = torch.randn(
        (batch, seqlen_q, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    prepare = {"BLOCK_N": kv_tile}
    k_descale = v_descale = 1.0
    if k_format in PACKED_K:
        k_quant = k_basis = k_real
        prepare.update(k_source=k_real, k_packed_format=k_format)
    elif k_format == "mxfp8":
        k_quant, prepare["k_scale"], k_basis = _mx(k_real)
    else:
        k_quant, k_descale = _per_tensor(k_real, k_format)
        k_basis = k_quant
    if v_format in PACKED_V:
        v_quant = v_basis = v_real
        prepare.update(v_source=v_real, v_packed_format=v_format)
    else:
        v_quant, v_descale = _per_tensor(v_real, v_format)
        v_basis = v_quant
    prep = sol_prepare(q, k_quant, v_quant, BETA, **prepare)
    return {
        "k": k_basis,
        "v": v_basis,
        "q_abs_max": sol_q_abs_max(q),
        "mean_k": prep["mean_k"],
        "mean_v": prep["mean_v"],
        "softmax_scale": softmax_scale,
        "rank": rank,
        "kv_tile": kv_tile,
        "k_format": k_format,
        "v_format": v_format,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "q_multiplier": q_multiplier,
        "mean_k_scale": prep["mean_k_scale"],
        "mean_v_scale": prep["mean_v_scale"],
        "mean_k_pooled": prep["mean_k_pooled"],
        "mean_v_pooled": prep["mean_v_pooled"],
    }


def _factor_kwargs(inputs, q_descale=1.0):
    """sol_tilt_factors's keyword arguments over make_tilt_inputs' e4m3 operands and Q maxima."""

    def one(x):
        return torch.full((1,), x, dtype=torch.float32, device="cuda")

    batch, _, heads, d = inputs["k"].shape
    seqlen_q = inputs["q_abs_max"][0].shape[1] * SOL_ATTN_TS_QO
    return {
        "q": torch.zeros((batch, seqlen_q, heads, d), dtype=FP8, device="cuda"),
        "k": inputs["k"],
        "v": inputs["v"],
        "q_descale": one(q_descale),
        "k_descale": one(inputs["k_descale"]),
        "v_descale": one(inputs["v_descale"]),
        "mean_k": inputs["mean_k"],
        "mean_v": inputs["mean_v"],
        "softmax_scale": inputs["softmax_scale"],
        "rank": inputs["rank"],
        "kv_tile": inputs["kv_tile"],
        "q_abs_max": inputs["q_abs_max"],
    }


def make_tilt_factor_inputs(q_descale=1.0, **kwargs):
    """sol_tilt_factors's keyword arguments for make_tilt_inputs("e4m3", "e4m3", **kwargs)."""
    return _factor_kwargs(make_tilt_inputs("e4m3", "e4m3", **kwargs), q_descale)


def _decoded_planes(planes, inputs):
    """The (rank, batch, blocks, heads, d) A and B planes, decoded as mean_k and mean_v are."""
    a, b = [], []
    for i in range(1, planes.rank + 1):
        a_i = _plane(planes.mean_k, planes.k_plane_stride, i).float()
        if inputs["k_format"] == "mxfp8":
            scale = _plane(planes.mean_k_scale, planes.k_scale_plane_stride, i)
            a_i = a_i * torch.exp2(scale.float() - 127.0).repeat_interleave(32, -1)
        a.append(a_i * inputs["k_descale"])
        b.append(
            _plane(planes.mean_v, planes.v_plane_stride, i).float()
            * inputs["v_descale"]
        )
    return torch.stack(a), torch.stack(b)


def _tilt_operator(planes, inputs):
    """Each block's (d_v, d_k) map from Q, as the kernel decodes Q, to the value update per P."""
    a, b = _decoded_planes(planes, inputs)
    units = planes.units.float()
    gain = units[2] * units[3] * inputs["q_multiplier"]
    return gain * torch.einsum("rbnhv,rbnhk->bnhvk", b, a)


def _reference_operator(inputs, block_gain=None):
    """softmax_scale / kv_tile * k_descale * v_descale times each full block's rank-r V_c^T K_c."""
    k, v, kv_tile = inputs["k"].float(), inputs["v"].float(), inputs["kv_tile"]
    full = k.shape[1] // kv_tile

    def blocks(x):
        x = x[:, : full * kv_tile].unflatten(1, (full, kv_tile))
        return x - x.mean(dim=2, keepdim=True)

    c = torch.einsum("bnthv,bnthk->bnhvk", blocks(v), blocks(k))
    u, s, vh = torch.linalg.svd(c)
    r = inputs["rank"]
    m = (u[..., :r] * s[..., None, :r]) @ vh[..., :r, :]
    m = m * (
        inputs["softmax_scale"] / kv_tile * inputs["k_descale"] * inputs["v_descale"]
    )
    if block_gain is not None:
        m = m * block_gain[:, :full, :, None, None]
    return m


# One plane's relative rounding error in each format; a block's operator rounds its A and B planes.
_PLANE_ROUNDING = {"e4m3": 0.03, "mxfp8": 0.03, "int8": 0.009, "bf16": 0.002}


def _assert_operator(got, want, k_format, v_format):
    """got is want to its planes' rounding, which is unbiased: their overall scale agrees closely."""
    rounding = math.hypot(_PLANE_ROUNDING[k_format], _PLANE_ROUNDING[v_format])
    error = (
        ((got - want).flatten(-2).norm(dim=-1) / want.flatten(-2).norm(dim=-1))
        .max()
        .item()
    )
    assert error < 1.6 * rounding, f"tilt operator relative error {error}"
    scale = ((got * want).sum() / (want * want).sum()).item()
    assert abs(scale - 1) < 0.012, f"tilt operator scale {scale}"


@pytest.mark.parametrize("v_format", ["e4m3", "bf16"])
@pytest.mark.parametrize("k_format", ["e4m3", "int8", "bf16", "mxfp8"])
def test_tilt_planes_decode_to_the_truncated_block_covariance(k_format, v_format):
    inputs = make_tilt_inputs(
        k_format, v_format, q_multiplier=2.0 if k_format == "mxfp8" else 1.0
    )
    # The operator is in real Q units, so Q's descale and multiplier cancel out of it.
    planes = sol_tilt_planes(**inputs, q_descale=0.5)
    full = inputs["k"].shape[1] // inputs["kv_tile"]

    got = _tilt_operator(planes, inputs)
    _assert_operator(got[:, :full], _reference_operator(inputs), k_format, v_format)
    # The short last block is computed exactly, so its planes hold no tilt.
    assert not got[:, full:].any()
    assert torch.equal(planes.mean_k, inputs["mean_k"])
    assert torch.equal(planes.mean_v, inputs["mean_v"])
    if k_format == "mxfp8":
        assert torch.equal(planes.mean_k_scale, inputs["mean_k_scale"])


def test_tilt_planes_scale_by_the_block_gain():
    inputs = make_tilt_inputs("e4m3", "bf16")
    blocks = inputs["mean_k"].shape[1]
    block_gain = torch.rand(
        (1, blocks, inputs["k"].shape[2]), dtype=torch.float32, device="cuda"
    )
    block_gain = 0.25 + 0.75 * block_gain
    planes = sol_tilt_planes(**inputs, block_gain=block_gain)
    full = inputs["k"].shape[1] // inputs["kv_tile"]
    _assert_operator(
        _tilt_operator(planes, inputs)[:, :full],
        _reference_operator(inputs, block_gain),
        "e4m3",
        "bf16",
    )


def test_tilt_planes_pad_each_mxfp8_k_scale_plane_to_a_whole_scale_tile():
    """Every MXFP8 K-scale plane is read 128 rows at a time, so each runs zeroed to that tile."""
    inputs = make_tilt_inputs("mxfp8", "e4m3", seqlen_k=2048, kv_tile=64)
    with _poisoned_allocations():
        planes = sol_tilt_planes(**inputs)
    scale, stride = planes.mean_k_scale, planes.k_scale_plane_stride
    blocks, row = scale.shape[1], scale[0, 0].numel()
    pad = (-blocks % SOL_ATTN_TS_KV) * row
    assert pad > 0 and stride >= scale.numel() + pad
    storage = _bytes(scale)
    assert storage.numel() >= planes.rank * stride + scale.numel() + pad
    for i in range(planes.rank + 1):
        start = i * stride + scale.numel()
        assert not storage[start : start + pad].any(), f"plane {i}"


@requires_gfx950
@pytest.mark.parametrize(
    "k_format, v_format",
    [
        ("mxfp6", "mxfp6_fp6_p"),
        ("mxfp6", "mxfp4_fp6_p"),
        ("mxfp4", "mxfp4_fp6_p"),
        ("e4m3", "mxfp6_fp6_p"),
    ],
)
def test_packed_tilt_planes_keep_plane_zero_and_the_packers_backing(k_format, v_format):
    """Plane 0 is the pooled operand packed again; the planes share one allocation, and the
    last plane keeps the zeroed backing its packer reserves for the reads past its end.
    """
    with _poisoned_allocations():
        inputs = make_tilt_inputs(k_format, v_format)
        planes = sol_tilt_planes(**inputs)
    assert torch.equal(planes.mean_k, inputs["mean_k"])
    assert torch.equal(planes.mean_v, inputs["mean_v"])
    pairs = [
        ("mean_v", planes.mean_v, planes.v_plane_stride),
        ("mean_v_scale", planes.mean_v_scale, planes.v_scale_plane_stride),
    ]
    if k_format in PACKED_K:
        pairs += [
            ("mean_k", planes.mean_k, planes.k_plane_stride),
            ("mean_k_scale", planes.mean_k_scale, planes.k_scale_plane_stride),
        ]
    for name, plane0, stride in pairs:
        packed = inputs[name]
        if name.endswith("_scale"):
            assert torch.equal(plane0, packed), name
        assert plane0.stride() == packed.stride(), name
        # Each plane spans what the packer lays out for one pooled batch: data, tile padding and
        # tails. After the last plane comes the packer's slack, zeroed where the packer zeroes it.
        assert stride == _plane_bytes(packed), name
        slack = _bytes(plane0)[(1 + planes.rank) * stride :]
        packed_slack = _bytes(packed)[stride:]
        assert slack.numel() == packed_slack.numel(), name
        assert not slack[packed_slack == 0].any(), name
        # The V and MXFP4 K scale readers run past their last row; the MXFP6 K reader takes its
        # scales from the records' tile tails, never from this image.
        if name == "mean_v_scale" or (name == "mean_k_scale" and k_format == "mxfp4"):
            assert packed_slack.numel() and not packed_slack.any(), name


def test_tilt_factors_share_the_e4m3_planes_factors():
    """sol_tilt_factors is the per-tensor FP8 encoding of the same factors.

    Its A planes are the e4m3 planes' codes, its B planes theirs times one multiplier, and its
    shift bytes follow _sol_tilt_shift_kernel's formula from the same peaks.
    """
    inputs = make_tilt_inputs("e4m3", "e4m3")
    q_descale = 0.75
    planes = sol_tilt_planes(**inputs, q_descale=q_descale)

    ext_k, ext_v, (rank, stride, shift) = sol_tilt_factors(
        **_factor_kwargs(inputs, q_descale)
    )
    assert (
        rank == planes.rank and stride == planes.k_plane_stride == planes.v_plane_stride
    )
    assert shift.dtype == torch.int32 and shift.shape == (1,)
    assert torch.equal(ext_k, inputs["mean_k"]) and torch.equal(ext_v, inputs["mean_v"])
    b_factors, b_planes = [], []
    for i in range(1, rank + 1):
        assert torch.equal(
            _plane(ext_k, stride, i), _plane(planes.mean_k, stride, i)
        ), f"A plane {i}"
        b_factors.append(_plane(ext_v, stride, i).float())
        b_planes.append(_plane(planes.mean_v, stride, i).float())
    b_factors, b_planes = torch.stack(b_factors), torch.stack(b_planes)

    # The peaks, recovered from the e4m3 planes' code multipliers.
    u0, u1 = planes.units[:2].tolist()
    a_top = FP8_MAX / u0
    b_top = FP8_MAX * sol_tilt._TILT_B_HEADROOM / u1
    qd, kd, vd = q_descale, inputs["k_descale"], inputs["v_descale"]
    softmax_scale, kv_tile = inputs["softmax_scale"], inputs["kv_tile"]
    base = sol_tilt._TILT_LOGIT_UNIT * softmax_scale * FP8_MAX * kd * qd / a_top
    score_shift = min(
        max(math.floor(math.log2(sol_tilt._TILT_W_MAX / base)), -128), 127
    )
    w_max = base * 2.0**score_shift
    b_scale = vd * kd * qd * softmax_scale / kv_tile
    value_shift = math.ceil(
        math.log2(b_top * b_scale / (vd * w_max * FP8_MAX * sol_tilt._TILT_B_HEADROOM))
    )
    value_shift = min(max(value_shift, -128), 127)
    word = shift.item()
    assert (word & 0xFF) == score_shift & 0xFF
    assert ((word >> 8) & 0xFF) == value_shift & 0xFF

    # One e4m3 rounding on each side of the multiplier.
    multiplier = b_scale / (vd * w_max * 2.0**value_shift) / u1
    fit = (b_factors * b_planes).sum() / (b_planes * b_planes).sum()
    assert math.isclose(fit.item(), multiplier, rel_tol=0.005)
    residual = (b_factors - fit * b_planes).norm() / b_factors.norm()
    assert residual.item() < 0.05


def _signed_byte(x):
    return x - 256 if x >= 128 else x


@pytest.mark.parametrize("strides", [(), (4096,), (4096, 8192, 12288)])
def test_tilt_valu_params_carry_the_value_shift_and_score_multiplier(strides):
    inputs = make_tilt_inputs("int8", "e4m3")
    planes = sol_tilt_planes(**inputs, q_descale=0.5)
    out = sol_tilt_valu_params(planes, 0.5, inputs["k_descale"], strides).tolist()
    assert len(out) == 2 + len(strides) and tuple(out[2:]) == strides
    vshift = _signed_byte((out[0] >> 8) & 0xFF)
    assert out[0] & ~0xFF00 == 0
    sigma = torch.tensor([out[1]], dtype=torch.int32).view(torch.float32).item()
    u2, u3 = planes.units[2:].tolist()
    qd_kd = 0.5 * inputs["k_descale"]
    assert math.isclose(sigma * 2.0**vshift, u2 * qd_kd * u3, rel_tol=1e-6)
    m = sigma / (u2 * qd_kd)
    assert sol_tilt._TILT_W_MAX / 2 < m <= sol_tilt._TILT_W_MAX * (1 + 1e-6)


def test_tilt_valu_params_reject_more_than_three_strides():
    planes = sol_tilt_planes(**make_tilt_inputs("int8", "e4m3"))
    with pytest.raises(ValueError, match="at most three strides"):
        sol_tilt_valu_params(planes, strides=(1, 2, 3, 4))


def test_tilt_scaled_params_carry_the_value_shift_and_score_multiplier():
    planes = sol_tilt_planes(**make_tilt_inputs("mxfp8", "e4m3", q_multiplier=2.0))
    out = sol_tilt_scaled_params(planes, planes.k_scale_plane_stride).tolist()
    assert len(out) == 3 and out[2] == planes.k_scale_plane_stride
    vshift = _signed_byte((out[0] >> 8) & 0xFF)
    sigma = torch.tensor([out[1]], dtype=torch.int32).view(torch.float32).item()
    u2, u3 = planes.units[2:].tolist()
    assert math.isclose(sigma * 2.0**vshift, u2 * u3, rel_tol=1e-6)
    assert sol_tilt._TILT_W_MAX / 2 < sigma / u2 <= sol_tilt._TILT_W_MAX * (1 + 1e-6)


def make_q_abs_max_inputs(kind, *, batch=2, seqlen=600, heads=3, q_tile=256):
    """(keyword arguments of sol_q_abs_max, Q as the kernel decodes it in fp32) for one Q kind.

    kind is "fp8" or "bf16" for a Q read as stored, "mxfp8" for E8M0-scaled codes, or "rotated"
    for a BF16 source Q that a packer takes through the hd128 rotation.
    """
    q = torch.randn(
        (batch, seqlen, heads, HEAD_DIM), dtype=torch.float32, device="cuda"
    )
    kwargs = {"q_tile": q_tile, "q_scale": None, "rotate": kind == "rotated"}
    if kind == "fp8":
        q = (q * 64.0).to(FP8)
        real = q.float()
    elif kind == "mxfp8":
        q, kwargs["q_scale"], real = _mx(q * 64.0)
        real = real.float()
    else:
        q = q.to(torch.bfloat16)
        real = q.float()
        if kind == "rotated":
            rotated = torch.empty_like(q)
            rotate_activation_hd128(rotated, q)
            real = rotated.float()
    return {"q": q, **kwargs}, real


@pytest.mark.parametrize("seqlen", [600, 1024])
@pytest.mark.parametrize("q_tile", [64, 256])
@pytest.mark.parametrize("kind", ["fp8", "bf16", "mxfp8", "rotated"])
def test_q_abs_max_is_the_tile_and_head_maximum(kind, q_tile, seqlen):
    kwargs, real = make_q_abs_max_inputs(kind, seqlen=seqlen, q_tile=q_tile)
    tiles, heads = sol_q_abs_max(**kwargs)

    pad = -seqlen % q_tile
    padded = torch.nn.functional.pad(real.abs(), (0, 0, 0, 0, 0, pad))
    want = padded.unflatten(1, (-1, q_tile)).amax(dim=2).to(torch.bfloat16)
    assert tiles.dtype == heads.dtype == torch.bfloat16
    if kind == "rotated":
        # The reference rotation rounds each value to bf16 before the maximum; the kernel after.
        torch.testing.assert_close(tiles, want, rtol=2**-7, atol=0)
    else:
        assert torch.equal(tiles, want)
    assert torch.equal(heads, tiles.amax(dim=1))


@pytest.mark.parametrize("kind", ["fp8", "bf16", "mxfp8", "rotated"])
def test_q_abs_max_compiles_fullgraph_and_matches_eager(kind):
    kwargs, _ = make_q_abs_max_inputs(kind, q_tile=64)
    _assert_same(
        sol_q_abs_max(**kwargs), _compiled(sol_q_abs_max)(**kwargs), "q_abs_max"
    )


def test_rotated_q_abs_max_first_called_in_a_graph_capture():
    """A capture records its kernels without running them, so the calls made before the graph
    first replays must not read anything that the capture was to make."""
    kwargs, real = make_q_abs_max_inputs("rotated", q_tile=64)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = sol_q_abs_max(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    eager = sol_q_abs_max(**kwargs)
    compiled = _compiled(sol_q_abs_max)(**kwargs)
    graph.replay()
    torch.cuda.synchronize()

    pad = -real.shape[1] % 64
    padded = torch.nn.functional.pad(real.abs(), (0, 0, 0, 0, 0, pad))
    want = padded.unflatten(1, (-1, 64)).amax(dim=2).to(torch.bfloat16)
    torch.testing.assert_close(eager[0], want, rtol=2**-7, atol=0)
    _assert_same(eager, compiled, "compiled", tails=False)
    _assert_same(eager, captured, "captured", tails=False)


def make_jensen_inputs(
    kind,
    *,
    batch=1,
    seqlen_q=700,
    seqlen_k=None,
    heads=2,
    kv_tile=SOL_ATTN_TS_KV,
    q_tile=SOL_ATTN_TS_QO,
    softmax_scale=0.125,
    seed=0,
):
    """(keyword arguments of sol_jensen_cap, sol_prepare's output) for one K kind.

    kind is "fp8" (per-tensor Q and K), "bf16", or the block-scaled "mxfp8", "mxfp4" or "mxfp6" K
    with a BF16 Q. K and V are planted as in _planted_kv, by default over 8 full blocks and a short
    one. The mask is sol_prepare's routed one at q_tile rows a tile.
    """
    seqlen_k = 8 * kv_tile + 21 if seqlen_k is None else seqlen_k
    k_real, v_real = _planted_kv(batch, seqlen_k, heads, 2, seed)
    q = 8.0 * torch.randn(
        (batch, seqlen_q, heads, HEAD_DIM), dtype=torch.float32, device="cuda"
    )
    prepare = {"BLOCK_M": q_tile, "BLOCK_N": kv_tile, "k_variance": True}
    cap = {"q_descale": None, "k_descale": None}
    if kind == "fp8":
        k, k_descale = _per_tensor(k_real, "e4m3")
        q, q_descale = _per_tensor(q, "e4m3")
        cap.update(
            q_descale=torch.full((1,), q_descale, dtype=torch.float32, device="cuda"),
            k_descale=torch.full((1,), k_descale, dtype=torch.float32, device="cuda"),
        )
    elif kind == "bf16":
        k, q = k_real, q.to(torch.bfloat16)
    elif kind == "mxfp8":
        k, prepare["k_scale"], _ = _mx(k_real)
        q = q.to(torch.bfloat16)
        cap["q_multiplier"] = 2.0
    else:
        k, q = k_real, q.to(torch.bfloat16)
        prepare.update(k_source=k_real, k_packed_format=kind)
        cap["q_multiplier"] = 2.0
    prep = sol_prepare(q, k, v_real, BETA, **prepare)
    if kind not in ("fp8", "bf16"):
        cap.update(
            mean_k_var_scale=prep["mean_k_var_scale"],
            mean_k_var_pooled=prep["mean_k_var_pooled"],
        )
    kwargs = {
        "q": q,
        "mean_k_var": prep["mean_k_var"],
        "softmax_scale": softmax_scale,
        "kv_tile": kv_tile,
        "block_attn_mask": prep["block_attn_mask"],
        "q_tile": q_tile,
        "q_abs_max": sol_q_abs_max(q, q_tile),
        **cap,
    }
    return kwargs, prep


def _jensen_reference(kind, kwargs, prep):
    """The uncapped per-block J_max in nats and the docstring's gain min(1, ln(kv_tile) / J_max)."""
    qm = kwargs["q_abs_max"][0].float()
    if kind in PACKED_K:
        var = prep["mean_k_var_pooled"].float()
    elif kind == "mxfp8":
        var = prep["mean_k_var"].float() * torch.exp2(
            prep["mean_k_var_scale"].float() - 127.0
        ).repeat_interleave(32, -1)
    else:
        var = prep["mean_k_var"].float()
    j = torch.einsum("bthd,bnhd->bhtn", qm * qm, var)
    if kwargs["block_attn_mask"] is not None:
        j = torch.where(kwargs["block_attn_mask"].bool(), 0.0, j)
    j_max = j.amax(dim=2).transpose(1, 2)
    softmax_scale = kwargs["softmax_scale"]
    if kind in ("fp8", "bf16"):
        per_nat = 0.5 * softmax_scale**2
        if prep["mean_k_var"].element_size() == 1:
            per_nat *= SOL_ATTN_FP8_VAR_DIVISOR
        if kwargs["q_descale"] is not None:
            per_nat *= (kwargs["q_descale"] * kwargs["k_descale"]).item() ** 2
    else:
        per_nat = 0.5 * (softmax_scale / kwargs["q_multiplier"]) ** 2
    j_top = j_max * per_nat
    gain = (math.log(kwargs["kv_tile"]) / j_top.clamp_min(1e-30)).clamp(max=1.0)
    return j_top, gain


def _cap_most(kind, kwargs, prep):
    """Set the softmax scale so that about two thirds of the pooled blocks are capped.

    The cap starts between the quartile and median blocks' terms, so no block sits on it.
    """
    kwargs["softmax_scale"] = 1.0
    j_top, _ = _jensen_reference(kind, kwargs, prep)
    live = j_top[j_top > 0]
    threshold = (live.quantile(0.25) + live.quantile(0.5)).item() / 2
    kwargs["softmax_scale"] = math.sqrt(math.log(kwargs["kv_tile"]) / threshold)


def _power_of_two_below(gain):
    return torch.exp2(torch.floor(torch.log2(gain.clamp(max=1.0))))


@pytest.mark.parametrize("masked", [False, True], ids=["all", "masked"])
@pytest.mark.parametrize("kv_tile", [64, SOL_ATTN_TS_KV])
@pytest.mark.parametrize(
    "kind",
    [
        "fp8",
        "bf16",
        "mxfp8",
        pytest.param("mxfp4", marks=requires_gfx950),
        pytest.param("mxfp6", marks=requires_gfx950),
    ],
)
def test_jensen_cap_damps_each_block_to_ln_kv_tile(kind, kv_tile, masked):
    kwargs, prep = make_jensen_inputs(kind, kv_tile=kv_tile)
    if not masked:
        kwargs["block_attn_mask"] = None
    _cap_most(kind, kwargs, prep)
    source = prep["mean_k_var"]
    source_before = source.clone()
    scale_before = (
        prep["mean_k_var_scale"].clone() if kind not in ("fp8", "bf16") else None
    )
    var, var_scale, gain = sol_jensen_cap(**kwargs)
    j_top, want = _jensen_reference(kind, kwargs, prep)
    assert (want < 1).any() and (want == 1).any()
    assert torch.equal(source, source_before)

    if kind in ("fp8", "bf16"):
        assert var_scale is None
        torch.testing.assert_close(gain, want, rtol=1e-5, atol=0)
        assert torch.equal(var, (source.float() * gain[..., None]).to(source.dtype))
        return

    # A block-scaled K takes the gain as a power of two in its scale bytes. A block whose gain
    # lies within the two sums' rounding of a step may take either side of it.
    ratio = math.log(kv_tile) / j_top.clamp_min(1e-30)
    near = (
        _power_of_two_below(ratio * (1 - 1e-5)),
        _power_of_two_below(ratio * (1 + 1e-5)),
    )
    assert ((gain == near[0]) | (gain == near[1])).all()
    assert (near[0] == near[1]).float().mean() > 0.9
    assert torch.equal(prep["mean_k_var_scale"], scale_before)
    shift = torch.log2(gain).to(torch.int32)[..., None]
    expected = scale_before.to(torch.int32)
    blocks = gain.shape[1]
    expected[:, :blocks] = (expected[:, :blocks] + shift).clamp(0, 254)
    assert torch.equal(var_scale.to(torch.int32), expected)
    if kind != "mxfp6":
        assert var is source
        return
    # The MXFP6 K reader takes its scales from each tile's tail in the records, so the cap
    # returns a copy of the records with the tails rebuilt from the new scale.
    rows, heads = var_scale.shape[1], var_scale.shape[2]
    tiles = -(-rows // 128)
    size = heads * tiles * fp6._K_TILE_BYTES
    records = source.as_strided((size,), (1,)).view(heads, tiles, fp6._K_TILE_BYTES)
    got = var.as_strided((size,), (1,)).view(heads, tiles, fp6._K_TILE_BYTES)
    tail = slice(fp6._K_SCALE_TAIL_OFFSET, fp6._K_SCALE_TAIL_OFFSET + 1024)
    index, valid = fp6._k_scale_tail_index(tiles, rows, heads, var_scale.device)
    want_tail = torch.where(valid, var_scale.reshape(-1)[index], 0)
    assert torch.equal(got[..., tail], want_tail)
    assert torch.equal(got[..., : tail.start], records[..., : tail.start])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("kv_tile", [64, SOL_ATTN_TS_KV])
@pytest.mark.parametrize(
    "k_format", ["mxfp8", pytest.param("mxfp4", marks=requires_gfx950)]
)
def test_jensen_cap_scale_keeps_the_pooled_scale_backing(k_format, kv_tile, compiled):
    """sol_jensen_cap's mean_k_var_scale is read as sol_prepare's is, past its last row.

    The cap writes a new scale, so its zeroed backing has to reach at least as far as the one it
    replaces: at 64x64 a 2048-token K pools to 32 rows, and the kernel still reads 128 of them.
    The scale bytes themselves are the input's moved by each block's exponent gain.
    """
    seqlen_k = 2048
    q = torch.randn((1, 512, 2, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((1, seqlen_k, 2, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((1, seqlen_k, 2, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    if k_format == "mxfp8":
        k_scale = (
            122 + torch.randint(0, 6, (*k.shape[:3], k.shape[3] // 32), device="cuda")
        ).to(torch.uint8)
        prep = sol_prepare(
            q, k.to(FP8), v.to(FP8), BETA, 64, kv_tile, k_scale=k_scale, k_variance=True
        )
    else:
        prep = sol_prepare(
            q,
            k,
            v,
            BETA,
            64,
            kv_tile,
            k_source=k,
            v_source=v,
            k_packed_format="mxfp4",
            v_packed_format="mxfp4_fp6_p",
            k_variance=True,
        )
    source = prep["mean_k_var_scale"]
    before = source.clone()
    cap = _compiled(sol_jensen_cap) if compiled else sol_jensen_cap
    _, scale, gain = cap(
        q,
        None,
        None,
        prep["mean_k_var"],
        0.125,
        kv_tile,
        q_abs_max=sol_q_abs_max(q),
        mean_k_var_scale=source,
        mean_k_var_pooled=prep["mean_k_var_pooled"],
    )

    assert torch.equal(source, before)
    assert scale.shape == source.shape and scale.stride() == source.stride()
    blocks = gain.shape[1]
    expected = before.to(torch.int32)
    shift = torch.log2(gain).round().to(torch.int32)[..., None]
    expected[:, :blocks] = (expected[:, :blocks] + shift).clamp(0, 254)
    assert torch.equal(scale.to(torch.int32), expected)
    tail = _allocated_tail(scale)
    assert tail.numel() >= _allocated_tail(source).numel()
    assert not tail.any()


def _assert_same(eager, compiled, name, tails=True):
    """Equal values, layouts and, with tails, the bytes past each tensor in its allocation."""
    if isinstance(eager, torch.Tensor):
        assert (
            compiled.shape == eager.shape and compiled.stride() == eager.stride()
        ), name
        assert torch.equal(compiled, eager), name
        if tails:
            assert torch.equal(_allocated_tail(compiled), _allocated_tail(eager)), name
    elif isinstance(eager, (tuple, list)):
        assert len(compiled) == len(eager), name
        for i, (e, c) in enumerate(zip(eager, compiled)):
            _assert_same(e, c, f"{name}[{i}]", tails)
    else:
        assert compiled == eager, name


def _compiled(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False)


@pytest.mark.parametrize(
    "k_format, v_format",
    [
        ("e4m3", "e4m3"),
        ("int8", "e4m3"),
        ("bf16", "bf16"),
        ("mxfp8", "e4m3"),
        pytest.param("mxfp6", "mxfp6_fp6_p", marks=requires_gfx950),
        pytest.param("mxfp4", "mxfp4_fp6_p", marks=requires_gfx950),
        pytest.param("e4m3", "mxfp6_fp6_p", marks=requires_gfx950),
    ],
)
def test_tilt_planes_compile_fullgraph_and_match_eager(k_format, v_format):
    inputs = make_tilt_inputs(k_format, v_format)
    eager = sol_tilt_planes(**inputs)
    compiled = _compiled(sol_tilt_planes)(**inputs)
    # A packer leaves its data slack as allocated, so only the planes themselves must agree.
    packed = k_format in PACKED_K or v_format in PACKED_V
    _assert_same(tuple(eager), tuple(compiled), "planes", tails=not packed)
    for name in ("k", "v", "k_scale", "v_scale"):
        view = getattr(eager, f"mean_{name}")
        if view is None:
            continue
        stride = getattr(eager, f"{name}_plane_stride")
        for i in range(1, eager.rank + 1):
            _assert_same(
                _plane(view, stride, i),
                _plane(getattr(compiled, f"mean_{name}"), stride, i),
                f"{name} plane {i}",
                tails=False,
            )
    _assert_same(
        sol_tilt_valu_params(eager, strides=(64,)),
        _compiled(sol_tilt_valu_params)(compiled, strides=(64,)),
        "valu",
    )
    _assert_same(
        sol_tilt_scaled_params(eager, 64),
        _compiled(sol_tilt_scaled_params)(compiled, 64),
        "scaled",
    )


def test_tilt_factors_compile_fullgraph_and_match_eager():
    kwargs = make_tilt_factor_inputs(q_descale=0.75)
    _assert_same(
        sol_tilt_factors(**kwargs), _compiled(sol_tilt_factors)(**kwargs), "factors"
    )


@pytest.mark.parametrize(
    "kind",
    [
        "fp8",
        "bf16",
        "mxfp8",
        pytest.param("mxfp4", marks=requires_gfx950),
        pytest.param("mxfp6", marks=requires_gfx950),
    ],
)
def test_jensen_cap_compiles_fullgraph_and_matches_eager(kind):
    kwargs, prep = make_jensen_inputs(kind)
    _cap_most(kind, kwargs, prep)
    _assert_same(sol_jensen_cap(**kwargs), _compiled(sol_jensen_cap)(**kwargs), "cap")


@pytest.mark.parametrize("path", ["factors", "planes"])
@pytest.mark.parametrize(
    "v_shape",
    [(1, 1024, 2, 128), (1, 4096, 2, 128), (2, 2048, 2, 128)],
    ids=["shorter", "longer", "batch"],
)
def test_tilt_rejects_v_not_shaped_as_k(path, v_shape):
    """The factor kernel loads V unmasked over K's extent, so a V shorter than K would be read
    past its allocation; any other V shape is refused before the launch."""
    kv_tile = SOL_ATTN_TS_KV
    k = torch.randn((1, 2048, 2, 128), dtype=torch.float32, device="cuda").to(FP8)
    v = torch.randn(v_shape, dtype=torch.float32, device="cuda").to(FP8)
    q = torch.randn((1, 512, 2, 128), dtype=torch.float32, device="cuda").to(FP8)
    mean = torch.zeros(
        (1, k.shape[1] // kv_tile, 2, 128), dtype=torch.float32, device="cuda"
    ).to(FP8)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="the tilt needs V shaped as K"):
        if path == "factors":
            sol_tilt_factors(q, k, v, one, one, one, mean, mean, 0.125, 2, kv_tile)
        else:
            sol_tilt_planes(
                k,
                v,
                sol_q_abs_max(q),
                mean,
                mean,
                0.125,
                2,
                kv_tile,
                k_format="e4m3",
                v_format="e4m3",
            )


@pytest.mark.parametrize("path", ["factors", "planes"])
@pytest.mark.parametrize(
    "kv_tile, k_blocks, v_blocks",
    [(64, 2, 2), (128, 4, 4), (128, 2, 1)],
    ids=["pooled_at_128_for_64", "pooled_at_64_for_128", "short_mean_v"],
)
def test_tilt_rejects_pooled_blocks_of_another_kv_tile(
    path, kv_tile, k_blocks, v_blocks
):
    """The factor kernel writes one plane row per full kv_tile block of K, and the pack kernel
    reads mean_v over mean_k's blocks, so pooled K/V of any other block count are refused before
    the planes are allocated."""
    k = torch.randn((1, 256, 1, 128), dtype=torch.float32, device="cuda").to(FP8)
    v = torch.randn((1, 256, 1, 128), dtype=torch.float32, device="cuda").to(FP8)
    q = torch.randn((1, 512, 1, 128), dtype=torch.float32, device="cuda").to(FP8)

    def pooled(blocks):
        rows = torch.zeros((1, blocks, 1, 128), dtype=torch.float32, device="cuda")
        return rows.to(FP8)

    mean_k, mean_v = pooled(k_blocks), pooled(v_blocks)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="pooled at another kv_tile"):
        if path == "factors":
            sol_tilt_factors(q, k, v, one, one, one, mean_k, mean_v, 0.125, 2, kv_tile)
        else:
            sol_tilt_planes(
                k,
                v,
                sol_q_abs_max(q),
                mean_k,
                mean_v,
                0.125,
                2,
                kv_tile,
                k_format="e4m3",
                v_format="e4m3",
            )

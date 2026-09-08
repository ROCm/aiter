import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    _get_config,
    fused_clamp_act_mul,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import unshuffle_scale_gemm
from aiter.utility import fp4_utils
from op_tests.triton_tests.quant.test_fused_fp8_quant import (
    per_token_fp8_group_quant,
    upcast,
)

_TORCH_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "relu6": lambda x: F.hardtanh(x, 0.0, 6.0),
}


def _check_backend(backend):
    """Run only the backend the card actually uses: gluon on gfx1250, triton
    everywhere else, so gfx1250 skips the triton cases and vice versa."""
    on_gfx1250 = get_arch() in ("gfx1250",)
    if backend == "gluon" and not on_gfx1250:
        pytest.skip("gluon backend requires gfx1250")
    if backend == "triton" and on_gfx1250:
        pytest.skip("gfx1250 runs the gluon backend only")


def _torch_reference(inp, swiglu_limit, weights, dtype_quant, activation="silu"):
    gate, up = inp.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    act_fn = _TORCH_ACTIVATIONS[activation]
    y = act_fn(gate) * up
    if weights is not None:
        y = weights.float() * y
    if dtype_quant is None:
        return y.to(inp.dtype)
    return per_token_fp8_group_quant(y, dtype_quant, 128)


@pytest.mark.parametrize("M", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("D", [2048, 3072])
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0])
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize(
    "with_weights,weight_broadcast",
    [(False, False), (True, True), (True, False)],
)
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul(
    M,
    D,
    swiglu_limit,
    transpose_scale,
    with_weights,
    weight_broadcast,
    dtype_quant,
    backend,
):

    _check_backend(backend)
    torch.manual_seed(42)
    N = D // 2
    if with_weights:
        if weight_broadcast:
            w = torch.randn(M, 1, device="cuda", dtype=torch.float32) * 0.5
        else:
            w = torch.randn(M, N, device="cuda", dtype=torch.float32) * 0.1
    else:
        w = None

    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    if dtype_quant is not None:
        out_buf = torch.empty((M, N), dtype=dtype_quant, device="cuda")
        if transpose_scale:
            scale = torch.empty(
                ((N + 127) // 128), M, dtype=torch.float32, device="cuda"
            )
        else:
            scale = torch.empty(
                (M, (N + 127) // 128), dtype=torch.float32, device="cuda"
            )

        out_q, scale = fused_clamp_act_mul(
            inp,
            out_buf,
            scale,
            swiglu_limit,
            weights=w,
            activation="silu",
            dtype_quant=dtype_quant,
            transpose_scale=transpose_scale,
            backend=backend,
        )

        ref_q, ref_s = _torch_reference(inp, swiglu_limit, w, dtype_quant)

        if transpose_scale:
            scale = scale.view(((N + 127) // 128), M).T.contiguous()
        out_triton = upcast(out_q, scale, torch.bfloat16)
        ref_triton = upcast(ref_q, ref_s, torch.bfloat16)

        torch.testing.assert_close(
            out_triton,
            ref_triton,
            atol=0.1,
            rtol=0.1,
        )
    else:
        # transpose_scale is irrelevant when not quantizing; skip the redundant
        # duplicate parametrization to keep the matrix small.
        if transpose_scale:
            pytest.skip("transpose_scale is only meaningful when dtype_quant is set")

        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=swiglu_limit,
            weights=w,
            activation="silu",
            dtype_quant=None,
            backend=backend,
        )
        ref = _torch_reference(inp, swiglu_limit, w, None)

        assert out.dtype == inp.dtype
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M, n_half", [(4096, 512), (8192, 512)])
@pytest.mark.parametrize("weight_broadcast", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_weights_multirow_tile(
    M, n_half, weight_broadcast, backend
):
    """Weights must be applied per row when a tile stages BLOCK_SIZE_M > 1 rows."""

    _check_backend(backend)

    config = _get_config(M, n_half, n_half, backend)
    block_m = config.get("BLOCK_SIZE_M", 1)
    if block_m <= 1:
        pytest.skip(f"shape stages one row per tile (BLOCK_SIZE_M={block_m})")

    torch.manual_seed(0)
    inp = torch.randn(M, 2 * n_half, device="cuda", dtype=torch.bfloat16)

    row_w = 1.0 + (torch.arange(M, device="cuda", dtype=torch.float32) % 4) * 0.5
    if weight_broadcast:
        w = row_w[:, None]
    else:
        col_w = torch.linspace(0.5, 1.5, n_half, device="cuda", dtype=torch.float32)
        w = row_w[:, None] * col_w[None, :]

    out = fused_clamp_act_mul(
        inp,
        swiglu_limit=0.0,
        weights=w,
        activation="silu",
        dtype_quant=None,
        backend=backend,
    )
    ref = _torch_reference(inp, 0.0, w, None)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M, n_half", [(4096, 512), (128, 1024)])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_broadcast_matches_expanded(M, n_half, backend):
    """A [M, 1] broadcast weight must equal the same values expanded to [M, N].

    The two take different code paths so testing if they match, and checked with
    torch.
    """
    _check_backend(backend)

    torch.manual_seed(3)
    inp = torch.randn(M, 2 * n_half, device="cuda", dtype=torch.bfloat16)
    w_col = torch.randn(M, 1, device="cuda", dtype=torch.float32) * 0.5

    out_broadcast = fused_clamp_act_mul(
        inp, swiglu_limit=0.0, weights=w_col, activation="silu", backend=backend
    )
    out_full = fused_clamp_act_mul(
        inp,
        swiglu_limit=0.0,
        weights=w_col.expand(M, n_half).contiguous(),
        activation="silu",
        backend=backend,
    )

    # check match
    torch.testing.assert_close(out_broadcast, out_full, atol=0.0, rtol=0.0)

    # check torch
    ref = _torch_reference(inp, 0.0, w_col, None)
    torch.testing.assert_close(out_broadcast, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_full, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M, n_half", [(4096, 512), (128, 1024)])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_full_weights_vary_along_n(M, n_half, backend):
    """Per-element weights must vary across N, not just across M.

    A kernel that collapsed [M, N] weights to one value per row would still
    pass a broadcast-only test, so every column has a distinct factor.
    """
    _check_backend(backend)

    torch.manual_seed(4)
    inp = torch.randn(M, 2 * n_half, device="cuda", dtype=torch.bfloat16)
    col_w = torch.linspace(0.25, 2.0, n_half, device="cuda", dtype=torch.float32)
    row_w = 1.0 + (torch.arange(M, device="cuda", dtype=torch.float32) % 4) * 0.5
    w = row_w[:, None] * col_w[None, :]

    out = fused_clamp_act_mul(
        inp, swiglu_limit=0.0, weights=w, activation="silu", backend=backend
    )
    ref = _torch_reference(inp, 0.0, w, None)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M, n_half", [(4096, 512)])
@pytest.mark.parametrize("weight_broadcast", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_weights_with_quant(M, n_half, weight_broadcast, backend):
    """Weights must be applied before the group scale is computed.

    The quant path derives the scale from the weighted values, so a weight
    indexing bug shows up in the scales as well, and this is
    the path the multi-row tuned configs are actually used on.
    """

    _check_backend(backend)

    torch.manual_seed(6)
    inp = torch.randn(M, 2 * n_half, device="cuda", dtype=torch.bfloat16)
    row_w = 1.0 + (torch.arange(M, device="cuda", dtype=torch.float32) % 4) * 0.5
    if weight_broadcast:
        w = row_w[:, None]
    else:
        col_w = torch.linspace(0.5, 1.5, n_half, device="cuda", dtype=torch.float32)
        w = row_w[:, None] * col_w[None, :]

    out_buf = torch.empty((M, n_half), dtype=aiter.dtypes.fp8, device="cuda")
    scale = torch.empty((M, (n_half + 127) // 128), dtype=torch.float32, device="cuda")
    out_q, scale = fused_clamp_act_mul(
        inp,
        out_buf,
        scale,
        0.0,
        weights=w,
        activation="silu",
        dtype_quant=aiter.dtypes.fp8,
        backend=backend,
    )
    ref_q, ref_s = _torch_reference(inp, 0.0, w, aiter.dtypes.fp8)

    torch.testing.assert_close(
        upcast(out_q, scale, torch.bfloat16),
        upcast(ref_q, ref_s, torch.bfloat16),
        atol=0.1,
        rtol=0.1,
    )


def _torch_reference_ue8m0(inp, swiglu_limit, weights, dtype_quant, quant_block_size):
    """Bit-exact torch model of the kernel's ue8m0 path: the exp2-based SiLU
    (matching ``_silu_exp2``) in fp32 followed by per-group MXFP8 quant with
    round-up e8m0 scales. Returns ``(out_q, unshuffled_scale)``."""
    gate, up = inp.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    y = (gate / (1.0 + torch.exp2(-(gate * 1.44269504089)))) * up
    if weights is not None:
        y = weights * y

    M, N = y.shape
    QB = quant_block_size
    dtype_max = torch.finfo(dtype_quant).max
    num_blocks = (N + QB - 1) // QB
    y = y.view(M, num_blocks, QB)
    max_val = y.abs().amax(dim=2, keepdim=True)
    dequant_scale = max_val / dtype_max
    # Round dequant_scale up to a power of two via the fp32 exponent field.
    exp = (dequant_scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000
    rounded = exp.view(torch.float32)
    quant_scale = torch.where(rounded == 0, torch.zeros_like(rounded), 1.0 / rounded)
    out_q = (y * quant_scale).view(M, N).to(dtype_quant)
    scale = (exp >> 23).to(torch.uint8).view(M, num_blocks)
    return out_q, scale


@pytest.mark.parametrize("M", [1, 2, 7, 32, 100, 257])
@pytest.mark.parametrize("D", [2048, 3072])
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0])
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("shuffle_scale", [False, True])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_ue8m0(
    M, D, swiglu_limit, with_weights, shuffle_scale, backend
):
    """ue8m0 group quant. The fp8 output and e8m0 scales must match the torch
    reference; when ``shuffle_scale`` is set the kernel must lay the scales out
    exactly like ``fp4_utils.e8m0_shuffle`` applied to the unshuffled scales."""

    _check_backend(backend)
    torch.manual_seed(42)
    N = D // 2
    quant_block_size = 32
    dtype_quant = torch.float8_e4m3fn
    w = (
        torch.randn(M, 1, device="cuda", dtype=torch.float32) * 0.5
        if with_weights
        else None
    )
    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    out_q, scale = fused_clamp_act_mul(
        inp,
        swiglu_limit=swiglu_limit,
        weights=w,
        activation="silu",
        dtype_quant=dtype_quant,
        quant_block_size=quant_block_size,
        scale_dtype_fmt="ue8m0",
        shuffle_scale=shuffle_scale,
        backend=backend,
    )

    ref_out, ref_scale = _torch_reference_ue8m0(
        inp, swiglu_limit, w, dtype_quant, quant_block_size
    )
    assert torch.equal(out_q.view(torch.uint8), ref_out.view(torch.uint8))

    num_blocks = (N + quant_block_size - 1) // quant_block_size
    if shuffle_scale:
        # Kernel preshuffles in place; the reference shuffles with e8m0_shuffle.
        # Both leave padding undefined, so undo the shuffle and compare the valid
        # region (which also confirms the kernel layout matches e8m0_shuffle).
        expected = fp4_utils.e8m0_shuffle(ref_scale)
        assert scale.shape == expected.shape
        sm = scale.shape[0]
        got = unshuffle_scale_gemm(scale.view(sm // 32, -1), arch="gfx950")[
            :M, :num_blocks
        ]
        exp = unshuffle_scale_gemm(expected.view(sm // 32, -1), arch="gfx950")[
            :M, :num_blocks
        ]
        assert torch.equal(got, exp)
        assert torch.equal(got, ref_scale)
    else:
        assert torch.equal(scale[:M, :num_blocks], ref_scale)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "relu", "relu6"])
@pytest.mark.parametrize("M", [1, 4, 32])
@pytest.mark.parametrize("D", [2048])
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_activations(M, D, activation, dtype_quant, backend):
    """Every activation path must match the torch reference."""
    _check_backend(backend)

    torch.manual_seed(42)
    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    if dtype_quant is not None:
        out_q, scale = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation=activation,
            dtype_quant=dtype_quant,
            backend=backend,
        )
        ref_q, ref_s = _torch_reference(inp, 0.0, None, dtype_quant, activation)
        torch.testing.assert_close(
            upcast(out_q, scale, torch.bfloat16),
            upcast(ref_q, ref_s, torch.bfloat16),
            atol=0.1,
            rtol=0.1,
        )
    else:
        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation=activation,
            dtype_quant=None,
            backend=backend,
        )
        ref = _torch_reference(inp, 0.0, None, None, activation)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "M, n_half",
    [
        # N=4096 config: M_LEQ_16384 → BLOCK_SIZE_M=2, ROWS_PER_PROG=2
        (16384, 4096),
        # N=8192 config: M_LEQ_8192 → BLOCK_SIZE_M=4, ROWS_PER_PROG=3
        (8192, 8192),
        # N=8192 config: M_LEQ_16384 → BLOCK_SIZE_M=2, ROWS_PER_PROG=1
        (16384, 8192),
    ],
)
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_large_n_configs(M, n_half, dtype_quant, backend):
    """Exercise tuned configs at N=4096 and N=8192 with ROWS_PER_PROG > 1."""
    _check_backend(backend)

    torch.manual_seed(7)
    inp = torch.randn(M, 2 * n_half, device="cuda", dtype=torch.bfloat16)

    if dtype_quant is not None:
        out_q, scale = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=dtype_quant,
            backend=backend,
        )
        ref_q, ref_s = _torch_reference(inp, 0.0, None, dtype_quant)
        torch.testing.assert_close(
            upcast(out_q, scale, torch.bfloat16),
            upcast(ref_q, ref_s, torch.bfloat16),
            atol=0.1,
            rtol=0.1,
        )
    else:
        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=None,
            backend=backend,
        )
        ref = _torch_reference(inp, 0.0, None, None)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M", [4, 32])
@pytest.mark.parametrize("D", [2048])
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
def test_fused_clamp_act_mul_backend_auto(M, D, dtype_quant):
    """backend=None must pick the right backend and produce correct results."""
    torch.manual_seed(42)
    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    if dtype_quant is not None:
        out_q, scale = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=dtype_quant,
            backend=None,
        )
        ref_q, ref_s = _torch_reference(inp, 0.0, None, dtype_quant)
        torch.testing.assert_close(
            upcast(out_q, scale, torch.bfloat16),
            upcast(ref_q, ref_s, torch.bfloat16),
            atol=0.1,
            rtol=0.1,
        )
    else:
        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=None,
            backend=None,
        )
        ref = _torch_reference(inp, 0.0, None, None)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# multiple rows per program according to config
@pytest.mark.parametrize(
    "M, n_half",
    [
        (13, 512),
        (4099, 512),
        (8195, 8192),
        (16387, 4096),
    ],
)
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_odd_m_tail(M, n_half, dtype_quant, backend):
    """Partial last tile: M not divisible by BLOCK_SIZE_M * ROWS_PER_PROG."""
    _check_backend(backend)

    torch.manual_seed(99)
    inp = torch.randn(M, 2 * n_half, device="cuda", dtype=torch.bfloat16)

    if dtype_quant is not None:
        out_q, scale = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=dtype_quant,
            backend=backend,
        )
        ref_q, ref_s = _torch_reference(inp, 0.0, None, dtype_quant)
        torch.testing.assert_close(
            upcast(out_q, scale, torch.bfloat16),
            upcast(ref_q, ref_s, torch.bfloat16),
            atol=0.1,
            rtol=0.1,
        )
    else:
        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=None,
            backend=backend,
        )
        ref = _torch_reference(inp, 0.0, None, None)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M", [1, 4, 32])
@pytest.mark.parametrize("D", [2048])
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_fused_clamp_act_mul_float16_input(M, D, dtype_quant, backend):
    """float16 inputs must work identically to bfloat16."""
    _check_backend(backend)

    torch.manual_seed(42)
    inp = torch.randn(M, D, device="cuda", dtype=torch.float16)

    if dtype_quant is not None:
        out_q, scale = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=dtype_quant,
            backend=backend,
        )
        ref_q, ref_s = _torch_reference(inp, 0.0, None, dtype_quant)
        torch.testing.assert_close(
            upcast(out_q, scale, torch.bfloat16),
            upcast(ref_q, ref_s, torch.bfloat16),
            atol=0.1,
            rtol=0.1,
        )
    else:
        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=0.0,
            activation="silu",
            dtype_quant=None,
            backend=backend,
        )
        ref = _torch_reference(inp, 0.0, None, None)
        assert out.dtype == inp.dtype
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

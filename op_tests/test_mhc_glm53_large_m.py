import pytest
import torch

from aiter.ops import mhc

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_SIZE = HC_MULT * (2 + HC_MULT)
RMS_EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_ITERS = 20


def _inputs(m: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    layer_input = (
        torch.randn(m, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) * 0.02
    )
    residual = (
        torch.randn(m, HC_MULT, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) * 0.02
    )
    post = torch.sigmoid(torch.randn(m, HC_MULT, device=device))
    comb = torch.softmax(torch.randn(m, HC_MULT, HC_MULT, device=device), dim=-1)
    fn = (
        torch.randn(
            MIX_SIZE,
            HC_MULT * HIDDEN_SIZE,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )
    scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    base = torch.zeros(MIX_SIZE, device=device, dtype=torch.float32)
    norm_weight = torch.linspace(
        0.75, 1.25, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    return layer_input, residual, post, comb, fn, scale, base, norm_weight


def _kwargs(norm_weight):
    return {
        "rms_eps": RMS_EPS,
        "hc_pre_eps": HC_EPS,
        "hc_sinkhorn_eps": HC_EPS,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": SINKHORN_ITERS,
        "norm_weight": norm_weight,
        "norm_eps": RMS_EPS,
    }


def _reference(layer_input, residual, post, comb, fn, scale, base, norm_weight):
    next_residual = torch.empty_like(residual)
    mhc.mhc_post(next_residual, layer_input, residual, post, comb)
    post_out, comb_out, layer_out = mhc.mhc_pre(
        next_residual, fn, scale, base, **_kwargs(norm_weight)
    )
    return post_out, comb_out, layer_out, next_residual


def _packed_reference(
    layer_input, residual, post, comb, packed_fn, scale, base, norm_weight
):
    next_residual = torch.empty_like(residual)
    mhc.mhc_post(next_residual, layer_input, residual, post, comb)
    post_out, comb_out, layer_out = mhc.mhc_pre(
        next_residual,
        packed_fn,
        scale,
        base,
        w_preshuffle_bf16=1,
        **_kwargs(norm_weight),
    )
    return post_out, comb_out, layer_out, next_residual


def _candidate(layer_input, residual, post, comb, packed_fn, scale, base, norm_weight):
    return mhc.mhc_fused_post_pre(
        layer_input,
        residual,
        post,
        comb,
        packed_fn,
        scale,
        base,
        force_fused=True,
        w_preshuffle_bf16=True,
        **_kwargs(norm_weight),
    )


def _assert_close(actual, expected):
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(actual[2], expected[2], atol=6.25e-2, rtol=3e-2)
    assert all(torch.isfinite(tensor).all() for tensor in actual)


@pytest.mark.parametrize("m", [4096, 8192, 16384, 131072])
@pytest.mark.parametrize("seed", [7, 29])
def test_glm53_prefill_fused_matches_unfused(m, seed):
    args = _inputs(m, seed)
    packed_fn = mhc.mhc_shuffle_fn(args[4])
    expected = _reference(*args)
    packed_expected = _packed_reference(*args[:4], packed_fn, *args[5:])
    actual = _candidate(*args[:4], packed_fn, *args[5:])
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    # The fused and unfused packed paths use the same RMSNorm finalizer. This
    # explicitly pins the final BF16 layer output while allowing the measured
    # MFMA accumulation-order difference.
    _assert_close(actual, packed_expected)


@pytest.mark.parametrize(
    "m,expected",
    [
        (4096, (4, 32, 32, 64)),
        (8192, (4, 32, 32, 64)),
        (16384, (2, 32, 32, 64)),
        (131072, (8, 32, 32, 64)),
    ],
)
def test_glm53_prefill_fused_config(m, expected):
    assert (
        mhc.get_mhc_fused_post_pre_config(
            m,
            HIDDEN_SIZE,
            w_preshuffle_bf16=True,
            res_preshuffle=False,
        )
        == expected
    )


@pytest.mark.parametrize("m", [1, 64, 512, 8192])
def test_glm53_fused_graph_replay(m):
    args = _inputs(m, 41)
    packed_fn = mhc.mhc_shuffle_fn(args[4])
    _candidate(*args[:4], packed_fn, *args[5:])
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _candidate(*args[:4], packed_fn, *args[5:])

    new_args = _inputs(m, 43)
    for static, new in zip(args[:4], new_args[:4]):
        static.copy_(new)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(*args[:4], args[4], *args[5:])
    _assert_close(captured, expected)

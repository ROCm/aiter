# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract of gemm_a8w8_bpreshuffle_out: it writes into a caller-owned bf16
[M, N] buffer, picks the same kernel as gemm_a8w8_bpreshuffle (so the result is
bit-identical on every backend), declares only ``out`` as mutated, and rejects a
buffer the kernels cannot write densely before any launch."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.ops import gemm_op_a8w8
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/HIP device"
)


def _gfx():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime()


def _inputs(m, n, k, w_k=None):
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    w = torch.randn((n, w_k or k), dtype=dtypes.bf16, device="cuda")
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    return xq, shuffle_weight(wq, layout=(16, 16)), x_scale, w_scale


def _backend(m, n, k):
    """Backend the tuned table selects for this shape, or None if untuned."""
    config = gemm_op_a8w8.get_GEMM_config_with_quant_type(
        m, n, k, dtypes.fp8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )
    if config is None:
        return None
    if "_8w_" in str(config.get("kernelName", "")):
        return "flydsl-8w"
    split = "-splitk" if int(config["splitK"]) > 0 else ""
    return config["libtype"] + split


def test_out_op_declares_only_out_mutable():
    schema = torch.ops.aiter.gemm_a8w8_bpreshuffle_out.default._schema
    written = [
        a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write
    ]
    assert written == ["out"]
    assert len(schema.returns) == 0


# One gfx950 tuned row per backend family the op can dispatch to.
_GFX950_ROWS = [
    ("ck", 1, 64, 4096),
    ("ck-splitk", 4, 64, 4096),
    ("cktile", 1, 1536, 128),
    ("flydsl", 1, 2048, 512),
    ("flydsl-splitk", 1, 576, 7168),
    ("flydsl-8w", 1856, 7168, 768),
]


@pytest.mark.skipif(
    torch.cuda.is_available() and _gfx() != "gfx950",
    reason="rows are picked from the gfx950 tuned tables",
)
@pytest.mark.parametrize(
    "backend,m,n,k", _GFX950_ROWS, ids=[r[0] for r in _GFX950_ROWS]
)
def test_out_matches_allocating_op_per_backend(backend, m, n, k):
    selected = _backend(m, n, k)
    assert selected == backend, (
        f"M={m} N={n} K={k} now selects {selected}, not {backend}; pick another "
        f"tuned row that selects {backend} so this backend stays covered"
    )
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16, device="cuda")

    assert aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale) is None
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "m,n,k,w_k",
    [(37, 1280, 512, None), (37, 1280, 448, 512)],
    ids=["untuned", "padded-k"],
)
def test_out_matches_allocating_op_untuned(m, n, k, w_k):
    for kk in {k, w_k or k}:
        assert _backend(m, n, kk) is None, (
            f"M={m} N={n} K={kk} is now tuned; pick an untuned shape so the "
            "default-kernel path stays covered"
        )
    xq, wq, x_scale, w_scale = _inputs(m, n, k, w_k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16, device="cuda")

    aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)
    assert torch.equal(out, ref)


def test_out_into_a_dense_view_of_a_larger_buffer():
    # The SGLang Kimi-K3 shared-down output is buf[offset:].view(M, N).
    m, n, k, offset = 37, 1280, 512, 37 * 512
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)
    flat = torch.full((offset + m * n,), 7.0, dtype=dtypes.bf16, device="cuda")
    out = flat[offset:].view(m, n)

    aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)
    assert torch.equal(out, ref)
    assert torch.all(flat[:offset] == 7.0)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_out_under_torch_compile(backend, monkeypatch):
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config, "force_disable_caches", True)
    m, n, k = 64, 1280, 512
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)

    def fn(out, xq, wq, x_scale, w_scale):
        aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)
        return out * 2

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16, device="cuda")
    res = compiled(out, xq, wq, x_scale, w_scale)
    assert torch.equal(out, ref)
    assert torch.equal(res, ref * 2)


@pytest.mark.parametrize(
    "make_out",
    [
        pytest.param(
            lambda m, n: torch.empty((m + 1, n), dtype=dtypes.bf16), id="shape"
        ),
        pytest.param(lambda m, n: torch.empty((m, n), dtype=dtypes.fp16), id="fp16"),
        pytest.param(lambda m, n: torch.empty((m, n), dtype=dtypes.fp32), id="fp32"),
        pytest.param(
            lambda m, n: torch.empty((m, 2 * n), dtype=dtypes.bf16)[:, :n],
            id="column-slice",
        ),
        pytest.param(
            lambda m, n: torch.empty((n, m), dtype=dtypes.bf16).t(), id="transposed"
        ),
    ],
)
def test_out_rejected_before_launch(make_out):
    m, n, k = 32, 256, 512
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    with torch.device("cuda"):
        out = make_out(m, n)
    storage = out._base if out._base is not None else out
    storage.view(torch.uint8).fill_(0xAB)

    with pytest.raises(ValueError, match="out must be"):
        aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)
    assert torch.all(storage.view(torch.uint8) == 0xAB)


@pytest.mark.parametrize("cpu_arg", ["WQ", "x_scale", "w_scale"])
def test_input_on_another_device_rejected_before_launch(cpu_arg):
    m, n, k = 32, 256, 512
    args = dict(zip(("XQ", "WQ", "x_scale", "w_scale"), _inputs(m, n, k)))
    args[cpu_arg] = args[cpu_arg].cpu()
    out = torch.full((m, n), 3.0, dtype=dtypes.bf16, device="cuda")

    with pytest.raises(ValueError, match="same device"):
        aiter.gemm_a8w8_bpreshuffle_out(out, *args.values())
    assert torch.all(out == 3.0)


def test_all_cpu_operands_rejected():
    m, n, k = 32, 256, 512
    args = [t.cpu() for t in _inputs(m, n, k)]
    out = torch.full((m, n), 3.0, dtype=dtypes.bf16)

    with pytest.raises(ValueError, match="CUDA/HIP tensor"):
        gemm_op_a8w8.gemm_a8w8_bpreshuffle_out(out, *args)
    assert torch.all(out == 3.0)


def test_weight_k_below_activation_k_rejected():
    m, n = 32, 256
    xq, _, x_scale, _ = _inputs(m, n, 512)
    _, wq, _, w_scale = _inputs(m, n, 448)
    out = torch.empty((m, n), dtype=dtypes.bf16, device="cuda")

    with pytest.raises(ValueError, match="WQ K >= XQ K"):
        aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)


@pytest.mark.parametrize("arg", ["XQ", "WQ"])
def test_non_2d_operand_rejected_before_launch(arg):
    # A rank-3 [M, 2, K/2] XQ keeps out at (M, N) and would otherwise dispatch on
    # K/2 while CK launches with K=2.
    m, n, k = 32, 256, 512
    args = dict(zip(("XQ", "WQ", "x_scale", "w_scale"), _inputs(m, n, k)))
    args[arg] = args[arg].view(args[arg].shape[0], 2, k // 2)
    out = torch.full((m, n), 3.0, dtype=dtypes.bf16, device="cuda")

    with pytest.raises(ValueError, match="must be 2-D"):
        aiter.gemm_a8w8_bpreshuffle_out(out, *args.values())
    assert torch.all(out == 3.0)

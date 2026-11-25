# SPDX-License-Identifier: MIT
"""Exercise route metadata caching without changing the reduction arithmetic."""

import functools

import pytest
import torch

from aiter.ops.flydsl.kernels import moe_gather_reduce as mod
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm device"
)


@functools.lru_cache(None)
def _builder(d, topk, dtype, sk, vec, wdtype, wave_routes=True):
    return mod.build_moe_gather_reduce_module(
        d, topk, dtype, sk, vec, wdtype, wave_routes=wave_routes
    )


def _run(x, ids, weights, out, valid, vec=2, wave_routes=True):
    sk, rows, d = x.shape
    dtype = "bf16" if x.dtype == torch.bfloat16 else "f16"
    w_dtype = {
        torch.float32: "f32",
        torch.bfloat16: "bf16",
        torch.float16: "f16",
    }[weights.dtype]
    fn = _builder(d, ids.shape[1], dtype, sk, vec, w_dtype, wave_routes)
    _run_compiled(
        fn,
        ptr_arg(x),
        ptr_arg(ids),
        ptr_arg(weights),
        ptr_arg(out),
        ids.shape[0],
        rows * (d // 2),
        ptr_arg(valid),
        current_stream_arg(),
    )
    return out


def _inputs(m, d, tk=5, sk=1, dtype=torch.bfloat16, wdtype=torch.float32):
    torch.manual_seed(411 + m + d)
    rows = max(16, m * max(tk, 1) + 1)
    x = (torch.randint(-4, 5, (sk, rows, d), device="cuda").float() / 16).to(dtype)
    ids = torch.randint(1, rows, (m, tk), device="cuda", dtype=torch.int32)
    weights = (torch.randint(0, 5, (m, tk), device="cuda").float() / 4).to(wdtype)
    if tk:
        ids[:, -1] = torch.arange(m, device="cuda", dtype=torch.int32) + 1
        weights[:, -1] = 1  # Shared expert: one contribution for every token.
        ids[::3, 0] = -1
    x[:, 0].fill_(float("nan"))  # Dropped routes must never read this row.
    return x, ids, weights


def _reference(x, ids, weights, valid):
    sk, _, d = x.shape
    m, tk = ids.shape
    total = torch.zeros((m, d), device=x.device, dtype=torch.float32)
    for k in range(tk):
        live = ids[:, k] >= 0
        row = ids[:, k].clamp_min(1).long()
        red = torch.zeros_like(total)
        for part in range(sk):
            red += torch.where(live[:, None], x[part, row].float(), 0)
        # Dyadic inputs and weights make these sums exact, including no-FMA
        # Torch evaluation vs a contracted device add/multiply.
        total += weights[:, k, None].float() * red
    out = torch.full((m, d), 83, device=x.device, dtype=x.dtype)
    out[:valid] = total[:valid].to(x.dtype)
    return out


@pytest.mark.parametrize("m", [1, 4, 12, 16, 32, 64, 128, 256, 2517])
def test_m3_shape(m):
    x, ids, weights = _inputs(m, 6144)
    out = torch.empty((m, 6144), device=x.device, dtype=x.dtype)
    valid = torch.tensor([m], device=x.device, dtype=torch.int32)
    _run(x, ids, weights, out, valid)
    torch.testing.assert_close(out, _reference(x, ids, weights, m), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("wdtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vec", [2, 4])
@pytest.mark.parametrize(
    "d,tk,sk", [(34, 1, 1), (126, 4, 2), (130, 5, 4), (6146, 8, 1), (66, 32, 2)]
)
def test_tails_and_splitk(dtype, wdtype, vec, d, tk, sk):
    x, ids, weights = _inputs(7, d, tk, sk, dtype, wdtype)
    out = torch.empty((7, d), device=x.device, dtype=dtype)
    valid = torch.tensor([7], device=x.device, dtype=torch.int32)
    for count in (7, 3, 0):
        valid.fill_(count)
        out.fill_(83)
        _run(x, ids, weights, out, valid, vec)
        torch.testing.assert_close(
            out, _reference(x, ids, weights, count), atol=0, rtol=0
        )


def test_graph_live_rows_and_streams():
    x, ids, weights = _inputs(17, 6144)
    out = torch.empty((17, 6144), device=x.device, dtype=x.dtype)
    valid = torch.tensor([17], device=x.device, dtype=torch.int32)
    _run(x, ids, weights, out, valid)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(x, ids, weights, out, valid)
    try:
        for count in (0, 13, 17, 1, 17):
            ids[::2, 0] = -1
            x.neg_()
            weights.neg_()
            valid.fill_(count)
            out.fill_(83)
            graph.replay()
            torch.testing.assert_close(
                out, _reference(x, ids, weights, count), atol=0, rtol=0
            )
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs = [torch.empty_like(out) for _ in streams]
        valid.fill_(17)
        for stream, dst in zip(streams, outputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _run(x, ids, weights, dst, valid)
        for stream in streams:
            stream.synchronize()
        for dst in outputs:
            torch.testing.assert_close(
                dst, _reference(x, ids, weights, 17), atol=0, rtol=0
            )
    finally:
        graph.reset()


def test_retained_lds_path():
    x, ids, weights = _inputs(3, 130)
    out = torch.empty((3, 130), device=x.device, dtype=x.dtype)
    valid = torch.tensor([3], device=x.device, dtype=torch.int32)
    _run(x, ids, weights, out, valid, wave_routes=False)
    torch.testing.assert_close(out, _reference(x, ids, weights, 3), atol=0, rtol=0)


def test_zero_topk_preserves_zero_output():
    x, ids, weights = _inputs(3, 34, tk=0)
    out = torch.full((3, 34), 83, device=x.device, dtype=x.dtype)
    valid = torch.tensor([3], device=x.device, dtype=torch.int32)
    _run(x, ids, weights, out, valid, wave_routes=False)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


def test_source_rows_beyond_four_gib():
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        pytest.skip("needs 6 GiB free for a sparse >4 GiB address-span test")
    d = 6144
    rows = 2**32 // (d * 2) + 10
    x = torch.empty((1, rows, d), device="cuda", dtype=torch.bfloat16)
    # Only touched rows are initialized. Indices include 64-bit base offsets
    # beyond 4 GiB; do not initialize the full tensor unnecessarily.
    selected = [1, rows - 2, rows - 1]
    for i, row in enumerate(selected):
        x[0, row].fill_((i + 1) / 4)
    ids = torch.tensor(
        [[selected[0], selected[1], selected[2], -1, selected[2]]],
        device="cuda",
        dtype=torch.int32,
    )
    weights = torch.ones((1, 5), device="cuda")
    out = torch.empty((1, d), device="cuda", dtype=x.dtype)
    valid = torch.tensor([1], device="cuda", dtype=torch.int32)
    _run(x, ids, weights, out, valid)
    torch.testing.assert_close(out, torch.full_like(out, 2.25), atol=0, rtol=0)


@pytest.mark.parametrize(
    "m,enabled",
    [
        (1, False),
        (3, False),
        (4, True),
        (16, True),
        (256, True),
        (257, False),
        (512, False),
    ],
)
def test_dispatch_boundaries(m, enabled, monkeypatch):
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    monkeypatch.setattr(grouped, "get_gfx_runtime", lambda: "gfx1250")
    assert grouped._use_wave_gather_routes(m, 6144, 5, "bf16", 1, "f32") == enabled


@pytest.mark.parametrize(
    "changed",
    [
        {"model_dim": 4096},
        {"topk": 4},
        {"out_dtype": "f16"},
        {"split_k": 2},
        {"w_dtype": "bf16"},
        {"w_dtype": "f16"},
    ],
)
def test_dispatch_other_contracts(changed, monkeypatch):
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    monkeypatch.setattr(grouped, "get_gfx_runtime", lambda: "gfx1250")
    args = {
        "token_num": 16,
        "model_dim": 6144,
        "topk": 5,
        "out_dtype": "bf16",
        "split_k": 1,
        "w_dtype": "f32",
    }
    args.update(changed)
    assert not grouped._use_wave_gather_routes(**args)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250-strict"])
def test_dispatch_other_arch(arch, monkeypatch):
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    monkeypatch.setattr(grouped, "get_gfx_runtime", lambda: arch)
    assert not grouped._use_wave_gather_routes(16, 6144, 5, "bf16", 1, "f32")

# SPDX-License-Identifier: MIT
"""Quantized-Q FP64 checks, metadata validation and changed-input graph replay."""

import math

import pytest
import torch

from aiter import sparse_mla_fp8_fwd, allocate_sparse_mla_fp8_workspace

QUERIES = list(range(1, 65)) + [65, 128, 256, 512, 1024, 2048]
CAPACITIES = [0, 1, 63, 64, 127, 128, 129, 2048, 2051, 2111, 2112, 2113, 2176]
TOLERANCE = 0.07

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx950 required",
)


def inputs(n, capacity=2112, *, strided=False, seed=2031071):
    torch.manual_seed(seed + n + capacity)
    shape = (n, 65, 520) if strided else (n, 64, 512)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)[:, :64, :512]
    kv = torch.randn((8192, 512), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    ids = torch.randint(len(kv), (n, capacity), device="cuda", dtype=torch.int32)
    if capacity:
        ids[0].fill_(-1)
        ids[:, 2051:] = -1
        if n > 1:
            ids[1, : min(9, capacity)] = len(kv) - 1
            ids[1, 10::97] = len(kv) + 17
    return q, kv, ids


def assert_reference(out, lse, q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16):
    """Independent FP64 attention after the documented scalar Q encoding."""
    decoded_q = (
        q.double() * q_scale
        if q.dtype == torch.float8_e4m3fn
        else (q.float() / q_scale).to(torch.float8_e4m3fn).double() * q_scale
    )
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    rows = sorted({0, min(1, len(q) - 1), len(q) // 2, len(q) - 1}) if len(q) else []
    worst = 0.0
    for row in rows:
        selected = ids[row].long()
        selected = selected[(selected >= 0) & (selected < len(kv))]
        if selected.numel():
            values = kv.view(torch.uint8)[selected].view(kv.dtype).double() * kv_scale
            scores = (decoded_q[row] @ values.T) * scale
            expected = scores.softmax(-1) @ values
            expected_lse = scores.logsumexp(-1)
        else:
            expected = torch.zeros((64, 512), device=q.device, dtype=torch.float64)
            expected_lse = torch.full(
                (64,), -math.inf, device=q.device, dtype=torch.float64
            )
        error = out[row].double() - expected
        rrms = error.square().mean(-1).sqrt() / expected.square().mean(
            -1
        ).sqrt().clamp_min(1e-3)
        worst = max(worst, rrms.max().item())
        assert rrms.max() <= TOLERANCE
        torch.testing.assert_close(
            out[row].double(), expected, atol=TOLERANCE, rtol=TOLERANCE
        )
        if lse is not None:
            torch.testing.assert_close(
                lse[row].double(), expected_lse, atol=TOLERANCE, rtol=TOLERANCE
            )
        if not selected.numel():
            assert torch.count_nonzero(out[row]) == 0
            if lse is not None:
                assert torch.isneginf(lse[row]).all()
    return worst


@pytest.mark.parametrize("n", QUERIES)
def test_quantized_reference(n):
    q, kv, ids = inputs(n)
    out, lse = sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
    assert_reference(out, lse, q, kv, ids)
    encoded = q.to(torch.float8_e4m3fn)
    control, control_lse = sparse_mla_fp8_fwd(encoded, kv, ids, 1 / 16, return_lse=True)
    assert torch.equal(out, control)
    assert torch.equal(lse, control_lse)


@pytest.mark.parametrize("n", [1, 8, 13, 24, 65, 129])
@pytest.mark.parametrize("capacity", CAPACITIES)
def test_capacity_stride_scale(n, capacity):
    q, kv, ids = inputs(n, capacity, strided=True)
    kwargs = dict(q_scale=0.5, kv_scale=1.25, return_lse=True)
    out, lse = sparse_mla_fp8_fwd(
        q, kv.view(-1, 1, 1, 512), ids[:, None, :], 1 / 16, **kwargs
    )
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)
    control, control_lse = sparse_mla_fp8_fwd(
        (q.float() / 0.5).to(torch.float8_e4m3fn), kv, ids, 1 / 16, **kwargs
    )
    assert torch.equal(out, control)
    assert torch.equal(lse, control_lse)


@pytest.mark.parametrize("n", [1, 6, 7, 8, 12, 13, 22, 24, 43, 65, 129, 2048])
def test_changed_graph(n):
    q, kv, ids = inputs(n)
    workspace = allocate_sparse_mla_fp8_workspace(q)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    kwargs = dict(out=out, lse=lse, return_lse=True, workspace=workspace)
    sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_lse = sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    pointers = [t.data_ptr() for t in [q, kv, ids, out, lse, *workspace]]
    for replay in range(4):
        q.normal_()
        fresh = torch.randn(kv.shape, device=kv.device, dtype=torch.bfloat16).to(
            kv.dtype
        )
        kv.copy_(fresh)
        ids.random_(0, len(kv))
        ids[:, 2051:] = -1
        if replay == 1:
            ids[:, :128] = len(kv) - 1
        elif replay == 2:
            ids[:, 3::11] = -1
        elif replay == 3:
            ids.fill_(-1)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        for t in workspace:
            t.fill_(math.nan)
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_lse = sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
        assert torch.equal(captured, expected)
        assert torch.equal(captured_lse, expected_lse)
        assert_reference(out, lse, q, kv, ids)
        assert pointers == [t.data_ptr() for t in [q, kv, ids, out, lse, *workspace]]


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("argument", ["q_scale", "kv_scale", "softmax_scale"])
def test_bad_scales(argument, scale):
    q, kv, ids = inputs(8, 1)
    kwargs = {"softmax_scale": 1 / 16, argument: scale}
    with pytest.raises((ValueError, RuntimeError), match="scale"):
        sparse_mla_fp8_fwd(q, kv, ids, **kwargs)


def test_metadata_and_storage():
    q, kv, ids = inputs(8, 1)
    with pytest.raises((ValueError, RuntimeError), match="dtype|FP8|float8"):
        sparse_mla_fp8_fwd(q, kv.to(torch.bfloat16), ids, 1 / 16)
    with pytest.raises((ValueError, RuntimeError), match="dtype"):
        sparse_mla_fp8_fwd(q, kv, ids.long(), 1 / 16)
    with pytest.raises((ValueError, RuntimeError), match="overlap"):
        sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, out=q)
    unaligned = torch.empty((q.numel() + 1,), device=q.device, dtype=q.dtype)[1:].view(
        q.shape
    )
    with pytest.raises((ValueError, RuntimeError), match="align"):
        sparse_mla_fp8_fwd(unaligned, kv, ids, 1 / 16)
    with pytest.raises((ValueError, RuntimeError), match="shape"):
        sparse_mla_fp8_fwd(
            q, kv, ids, 1 / 16, workspace=allocate_sparse_mla_fp8_workspace(q[:1])
        )
    oversized = torch.empty((8, 2177), device=q.device, dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError), match="2176"):
        sparse_mla_fp8_fwd(q, kv, oversized, 1 / 16)
    empty, empty_lse = sparse_mla_fp8_fwd(q[:0], kv, ids[:0], 1 / 16, return_lse=True)
    assert empty.shape == (0, 64, 512) and empty_lse.shape == (0, 64)


def test_torch_compile():
    q, kv, ids = inputs(8, 129)

    def run(q, kv, ids):
        return sparse_mla_fp8_fwd(q, kv, ids, 1 / 16)[0]

    expected = run(q, kv, ids)
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    assert torch.equal(compiled(q, kv, ids), expected)

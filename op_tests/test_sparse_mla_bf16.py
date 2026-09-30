# SPDX-License-Identifier: MIT
"""BF16 D512 sparse MLA: independent FP64, ragged CSR, strides and graph replay.

Run: python -m pytest -q op_tests/test_sparse_mla_bf16.py
"""

import pytest
import torch

from aiter import sparse_mla_bf16_fwd
from aiter.jit.utils.chip_info import get_gfx_runtime

pytestmark = pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="requires gfx950")
SPLITS = [1, 2, 4, 8, 16, 32]


def reference(q, kv, indptr, indices, scale):
    """Reference the mathematical operation, independently of tile/split layout."""
    output = torch.zeros_like(q, dtype=torch.float64)
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device, dtype=torch.float64)
    offsets = indptr.cpu().tolist()
    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        selected = indices[start:end].long()
        selected = selected[(selected >= 0) & (selected < kv.shape[0])]
        if not selected.numel():
            continue
        values = kv[selected].double()
        scores = q[row].double() @ values.T * scale
        output[row] = scores.softmax(-1) @ values
        lse[row] = scores.logsumexp(-1)
    return output, lse


def check(actual, expected):
    out, lse = actual
    ref_o, ref_lse = expected
    torch.testing.assert_close(out.double(), ref_o, atol=1e-2, rtol=1e-2)
    rms = (out.double() - ref_o).square().mean(-1).sqrt()
    relative = rms / ref_o.square().mean(-1).sqrt().clamp_min(1e-3)
    assert (relative <= 1e-2).all(), relative.max().item()
    if lse is not None:
        torch.testing.assert_close(lse.double(), ref_lse, atol=1e-4, rtol=1e-5)
    return relative.max().item() if relative.numel() else 0.0


def fixture(heads, topk, padded=False):
    torch.manual_seed(321 + heads + topk)
    queries, slots = 4, max(65, topk + 7)
    if padded:
        # Offset by one BF16, with padded query/head/cache rows.
        q_store = torch.randn(queries, heads, 520, device="cuda", dtype=torch.bfloat16)
        kv_store = torch.randn(slots, 520, device="cuda", dtype=torch.bfloat16)
        q, kv = q_store[..., 1:513], kv_store[:, 1:513]
    else:
        q = torch.randn(queries, heads, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(slots, 512, device="cuda", dtype=torch.bfloat16)
    lengths = [topk, max(0, topk - 3), 0, int(topk > 0)]
    ptr = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    idx = torch.randint(slots, (sum(lengths),), device="cuda", dtype=torch.int32)
    if topk > 1:
        idx[:topk:11] = -1
        idx[1:topk:17] = slots + 1
    # Duplicate rows intentionally retain multiplicity.
    if topk > 4:
        idx[2:4] = 3
    return q, kv, ptr, idx


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("splits", SPLITS)
@pytest.mark.parametrize("topk", [0, 1, 31, 32, 33, 63, 65, 2048, 2051])
@pytest.mark.parametrize("padded", [False, True])
def test_fp64(version, heads, splits, topk, padded):
    q, kv, ptr, idx = fixture(heads, topk, padded)
    q_count = q.shape[0]
    out_storage = torch.full((q.numel() + 2,), 123, device=q.device, dtype=q.dtype)
    out = out_storage[1:-1].view(q.shape)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    workspace = None
    if splits > 1:
        count = q_count * splits * heads
        # Four-byte-aligned partial O exercises the scalar fallback.
        partial = torch.empty(count * 512 + 1, device=q.device, dtype=torch.float32)[1:]
        workspace = (
            partial.view(q_count, splits, heads, 512),
            torch.empty((q_count, splits, heads), device=q.device),
        )
    result = sparse_mla_bf16_fwd(
        q,
        kv,
        ptr,
        idx,
        1 / 16,
        version=version,
        kv_splits=splits,
        out=out,
        lse=lse,
        return_lse=True,
        workspace=workspace,
    )
    assert result[0].data_ptr() == out.data_ptr()
    assert result[1].data_ptr() == lse.data_ptr()
    check(result, reference(q, kv, ptr, idx, 1 / 16))
    assert out_storage[0] == 123 and out_storage[-1] == 123


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("splits", SPLITS)
@pytest.mark.parametrize("return_lse", [False, True])
def test_graph_current_stream(version, heads, splits, return_lse):
    q, kv, ptr, idx = fixture(heads, 65)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], device=q.device) if return_lse else None
    workspace = (
        (
            torch.empty((4, splits, heads, 512), device=q.device),
            torch.empty((4, splits, heads), device=q.device),
        )
        if splits > 1
        else None
    )

    def run():
        return sparse_mla_bf16_fwd(
            q,
            kv,
            ptr,
            idx,
            1 / 16,
            version=version,
            kv_splits=splits,
            out=out,
            lse=lse,
            return_lse=return_lse,
            workspace=workspace,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        # Replay must consume new device values, not captured query/index data.
        q.mul_(0.5)
        idx.copy_(idx.roll(1))
        for _ in range(30):
            graph.replay()
        expected = reference(q, kv, ptr, idx, 1 / 16)
    torch.cuda.current_stream().wait_stream(stream)
    check((out, lse), expected)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("splits", [1, 32])
def test_subnormal_single_key(version, heads, splits):
    q = torch.zeros((1, heads, 512), device="cuda", dtype=torch.bfloat16)
    bits = torch.tensor(
        [1, 2, 127, -32767, -32766, -32641, 0, 0], device="cuda", dtype=torch.int16
    )
    kv = bits.repeat(64).view(torch.bfloat16).view(1, 512)
    ptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    idx = torch.zeros(1, device="cuda", dtype=torch.int32)
    out, lse = sparse_mla_bf16_fwd(
        q, kv, ptr, idx, 1 / 16, version=version, kv_splits=splits, return_lse=True
    )
    assert torch.equal(
        out.view(torch.int16), kv.expand(heads, 512)[None].view(torch.int16)
    )
    assert (lse == 0).all()


@pytest.mark.parametrize("splits", SPLITS)
@pytest.mark.parametrize("above_boundary", [False, True])
def test_h64_hybrid_dispatch_boundary(splits, above_boundary):
    # Q*S=128 uses head tiles; the next query switches to the four-wave main.
    # The small ragged fixture alone stays on the head-tiled branch for all S.
    queries = 128 // splits + int(above_boundary)
    torch.manual_seed(812)
    q = torch.randn((queries, 64, 512), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((97, 512), dtype=torch.bfloat16, device="cuda")
    ptr = torch.arange(queries + 1, dtype=torch.int32, device="cuda") * 65
    idx = torch.randint(97, (queries, 65), dtype=torch.int32, device="cuda")
    idx[::2, 16:32] = -1
    idx[-1] = -1
    result = sparse_mla_bf16_fwd(
        q,
        kv,
        ptr,
        idx.flatten(),
        1 / 16,
        version=2,
        kv_splits=splits,
        return_lse=True,
    )
    check(result, reference(q, kv, ptr, idx.flatten(), 1 / 16))


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("pool_shape", ["flat", "paged", "mla"])
def test_pool_views_empty_q_and_compile(version, pool_shape):
    q, kv, ptr, idx = fixture(16, 33)
    # Slice to a multiple of the page size; invalid ids are masked by contract.
    kv = kv[:64]
    pool = {"flat": kv, "paged": kv.view(8, 8, 512), "mla": kv[:, None, None]}[
        pool_shape
    ]
    run = torch.compile(sparse_mla_bf16_fwd, backend="eager", fullgraph=True)
    actual = run(
        q, pool, ptr, idx, 1 / 16, version=version, kv_splits=2, return_lse=True
    )
    check(actual, reference(q, kv, ptr, idx, 1 / 16))
    empty = sparse_mla_bf16_fwd(
        q[:0],
        pool,
        ptr[:1],
        idx[:0],
        1 / 16,
        version=version,
        kv_splits=2,
        return_lse=True,
    )
    assert empty[0].shape == (0, 16, 512) and empty[1].shape == (0, 16)


@pytest.mark.parametrize(
    "change",
    ["dtype", "width", "heads", "splits", "version", "scale", "alias", "workspace"],
)
def test_reject_invalid_metadata(change):
    q, kv, ptr, idx = fixture(16, 33)
    args = dict(kv_splits=2)
    if change == "dtype":
        q = q.float()
    elif change == "width":
        q = q[..., :256]
    elif change == "heads":
        q = q[:, :8]
    elif change == "splits":
        args["kv_splits"] = 3
    elif change == "version":
        args["version"] = 3
    elif change == "alias":
        args["out"] = q
    elif change == "workspace":
        args["workspace"] = (
            torch.empty(1, device=q.device),
            torch.empty(1, device=q.device),
        )
    with pytest.raises((RuntimeError, ValueError)):
        sparse_mla_bf16_fwd(
            q, kv, ptr, idx, float("nan") if change == "scale" else 1 / 16, **args
        )

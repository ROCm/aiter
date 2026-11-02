# SPDX-License-Identifier: MIT
"""Quantized-Q FP64 checks, metadata validation and changed-input graph replay."""

import math

import pytest
import torch

from aiter import allocate_sparse_mla_fp8_workspace, sparse_mla_fp8_fwd

QUERIES = list(range(1, 65)) + [
    65,
    128,
    256,
    512,
    1024,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16383,
    16384,
]
CAPACITIES = [0, 1, 63, 64, 127, 128, 129, 2048, 2051, 2111, 2112, 2113, 2176]
TOLERANCE = 0.07

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx950 required",
)


def inputs(n, capacity=2112, *, strided=False, seed=2031071, padded=False):
    torch.manual_seed(seed + n + capacity)
    shape = (n, 65, 520) if strided else (n, 64, 512)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)[:, :64, :512]
    kv = torch.randn((8192, 512), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    ids = torch.randint(len(kv), (n, capacity), device="cuda", dtype=torch.int32)
    if capacity:
        if n > 1:
            ids[0].fill_(-1)
        if padded:
            ids[:, 2051:] = -1
        if n > 1:
            ids[1, : min(9, capacity)] = len(kv) - 1
            ids[1, 10::97] = len(kv) + 17
    return q, kv, ids


def reference_rows(n):
    """All small-Q rows; stratified rows and both sides of 2K boundaries otherwise."""
    if n <= 129:
        return list(range(n))
    rows = {0, 1, n - 1}
    rows.update(i * (n - 1) // 31 for i in range(32))
    for boundary in range(2048, n, 2048):
        rows.update([boundary - 1, boundary])
    return sorted(rows)


def assert_reference(
    out, lse, q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16, all_rows=False
):
    """Independent ideal FP64 attention after scalar Q encoding; bounded batches."""
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    rows = list(range(len(q))) if all_rows else reference_rows(len(q))
    worst = 0.0
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        batch_scale = (
            q_scale[batch]
            if isinstance(q_scale, torch.Tensor) and q_scale.ndim == 3
            else q_scale
        )
        encoded = (
            q[batch]
            if q.dtype == torch.float8_e4m3fn
            else (q[batch].float() / batch_scale).to(torch.float8_e4m3fn)
        )
        decoded = encoded.double() * batch_scale
        selected = ids[batch].long()
        valid = (selected >= 0) & (selected < len(kv))
        values = (
            kv.view(torch.uint8)[selected.clamp(0, len(kv) - 1)].view(kv.dtype).double()
            * kv_scale
        )
        values[~valid] = 0
        scores = torch.bmm(decoded, values.transpose(1, 2)) * scale
        scores.masked_fill_(~valid[:, None, :], -math.inf)
        expected_lse = scores.logsumexp(-1)
        nonempty = valid.any(-1)
        probability = torch.where(nonempty[:, None, None], scores.softmax(-1), 0.0)
        expected = torch.bmm(probability, values)
        error = out[batch].double() - expected
        rrms = error.square().mean(-1).sqrt() / expected.square().mean(
            -1
        ).sqrt().clamp_min(1e-3)
        worst = max(worst, rrms.max().item())
        assert rrms.max() <= TOLERANCE
        torch.testing.assert_close(
            out[batch].double(), expected, atol=TOLERANCE, rtol=TOLERANCE
        )
        if lse is not None:
            torch.testing.assert_close(
                lse[batch].double(), expected_lse, atol=5e-4, rtol=1e-5
            )
        assert torch.count_nonzero(out[batch][~nonempty]) == 0
        if lse is not None:
            assert torch.isneginf(lse[batch][~nonempty]).all()
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
@pytest.mark.parametrize("padded", [False, True])
def test_capacity_stride_scale(n, capacity, padded):
    q, kv, ids = inputs(n, capacity, strided=True, padded=padded)
    kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
    out, lse = sparse_mla_fp8_fwd(
        q, kv.view(-1, 1, 1, 512), ids[:, None, :], 1 / 16, **kwargs
    )
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)
    control, control_lse = sparse_mla_fp8_fwd(
        (q.float() / 0.5).to(torch.float8_e4m3fn), kv, ids, 1 / 16, **kwargs
    )
    assert torch.equal(out, control)
    assert torch.equal(lse, control_lse)
    # The benchmark also supplies per-row/head decoding scales for Gluon.
    # Cover that reference API across batch boundaries, with encoded Q.
    scales = torch.full((*q.shape[:2], 1), 0.5, device=q.device)
    assert_reference(
        control,
        control_lse,
        (q.float() / 0.5).to(torch.float8_e4m3fn),
        kv,
        ids,
        q_scale=scales,
        kv_scale=1.25,
    )


@pytest.mark.parametrize(
    "n", [1, 6, 7, 8, 12, 13, 22, 24, 43, 65, 129, 2048, 4096, 8192, 16384]
)
def test_changed_graph(n):
    q, kv, ids = inputs(n)
    workspace = allocate_sparse_mla_fp8_workspace(q)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    kwargs = {"out": out, "lse": lse, "return_lse": True, "workspace": workspace}
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
        if replay == 0:
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


@pytest.mark.parametrize("n", [4096, 8192, 16384])
def test_large_query_chunk_equivalence(n):
    """Every row agrees with the previously supported S1 launch size."""
    q, kv, ids = inputs(n, 2176, strided=True)
    kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
    out, lse = sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25, all_rows=True)
    for start in range(0, n, 2048):
        chunk_out, chunk_lse = sparse_mla_fp8_fwd(
            q[start : start + 2048], kv, ids[start : start + 2048], 1 / 16, **kwargs
        )
        assert torch.equal(out[start : start + 2048], chunk_out)
        assert torch.equal(lse[start : start + 2048], chunk_lse)


# Each public dispatch family, including compact Q8 and both streaming modes.
DISPATCH_QUERIES = [1, 6, 7, 8, 12, 13, 21, 22, 23, 24, 42, 43, 64, 65, 128, 129]


def arithmetic_inputs(n, capacity, *, seed):
    """Dyadic Q/KV keep QK sums exact, isolating P and partial-store rounding.

    General normal-distributed inputs still use the ideal FP64 check separately.
    """
    q, kv, ids = inputs(n, capacity, strided=True, seed=seed)
    q.copy_((torch.randint(-8, 9, q.shape, device=q.device).float() / 8).to(q.dtype))
    kv.copy_(
        (torch.randint(-8, 9, kv.shape, device=kv.device).float() / 8).to(kv.dtype)
    )
    return q, kv, ids


def arithmetic_reference(q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16):
    """Torch FP64 matmuls with explicit FP8 P and BF16 split-store rounding.

    This calls no AITER kernel or production helper and uses no device layouts.
    Keep the ideal FP64 reference above as a separate quantization-error check.
    """
    splits = next(
        s
        for limit, s in [
            (6, 17),
            (12, 9),
            (21, 6),
            (23, 4),
            (42, 6),
            (64, 4),
            (128, 2),
            (16384, 1),
        ]
        if len(q) <= limit
    )
    outputs, lses = [], []
    for row in range(len(q)):
        encoded = (
            q[row]
            if q.dtype == torch.float8_e4m3fn
            else (q[row].float() / q_scale).to(torch.float8_e4m3fn)
        )
        decoded = encoded.double() * q_scale
        selected = ids[row].long()
        valid = (selected >= 0) & (selected < len(kv))
        values = kv.view(torch.uint8)[selected.clamp(0, len(kv) - 1)].view(kv.dtype)
        values = values.double() * kv_scale
        values[~valid] = 0
        scores = (decoded @ values.T) * scale
        scores[:, ~valid] = -math.inf
        parts, part_lses = [], []
        tile_count = (len(selected) + 127) // 128
        quotient, remainder = divmod(tile_count, splits)
        begin = 0
        for split in range(splits):
            end = begin + quotient + (split < remainder)
            maximum = torch.full((64,), -math.inf, device=q.device, dtype=torch.float64)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((64, 512), device=q.device, dtype=torch.float64)
            for tile in range(begin, end):
                lo, hi = tile * 128, min((tile + 1) * 128, len(selected))
                block = scores[:, lo:hi]
                new_maximum = torch.maximum(maximum, block.amax(-1))
                alpha = torch.where(
                    torch.isfinite(maximum), (maximum - new_maximum).exp(), 0.0
                )
                probability = torch.where(
                    torch.isfinite(new_maximum[:, None]),
                    (block - new_maximum[:, None]).exp(),
                    0.0,
                )
                rounded_p = (probability.float() * 256).to(
                    torch.float8_e4m3fn
                ).double() / 256
                numerator = numerator * alpha[:, None] + rounded_p @ values[lo:hi]
                denominator = denominator * alpha + probability.sum(-1)
                maximum = new_maximum
            partial = numerator / denominator.clamp_min(1e-300)[:, None]
            if splits > 1:
                partial = partial.to(torch.bfloat16).double()
            parts.append(partial)
            part_lses.append(maximum + denominator.log())
            begin = end
        logs = torch.stack(part_lses)
        lse = logs.logsumexp(0)
        weights = torch.where(torch.isfinite(lse)[None, :], (logs - lse).exp(), 0.0)
        out = (torch.stack(parts) * weights[:, :, None]).sum(0)
        outputs.append(out.to(torch.bfloat16))
        lses.append(lse)
    return torch.stack(outputs), torch.stack(lses)


def assert_arithmetic_reference(out, lse, expected, expected_lse):
    error = out.double() - expected.double()
    rrms = error.square().mean(-1).sqrt() / expected.double().square().mean(
        -1
    ).sqrt().clamp_min(1e-3)
    assert rrms.max() <= 0.005
    torch.testing.assert_close(out.double(), expected.double(), atol=0.002, rtol=0.005)
    torch.testing.assert_close(lse.double(), expected_lse, atol=1e-5, rtol=1e-6)
    return rrms.max().item()


@pytest.mark.parametrize("n", DISPATCH_QUERIES)
@pytest.mark.parametrize("capacity", [2112, 2176])
@pytest.mark.parametrize("tail_only", [False, True])
@pytest.mark.parametrize("seed", [38119, 38123])
def test_arithmetic_reference_valid_tail(n, capacity, tail_only, seed, record_property):
    q, kv, ids = arithmetic_inputs(n, capacity, seed=seed)
    ids[:, 2051:] = torch.randint(
        len(kv), (n, capacity - 2051), device=q.device, dtype=torch.int32
    )
    if tail_only:
        ids[:, :2051] = -1
    # Every tail position is a valid slot, including the last API entry.
    assert ((ids[:, 2051:] >= 0) & (ids[:, 2051:] < len(kv))).all()
    kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
    out, lse = sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    expected, expected_lse = arithmetic_reference(
        q, kv, ids, q_scale=0.5, kv_scale=1.25
    )
    worst = assert_arithmetic_reference(out, lse, expected, expected_lse)
    record_property("arithmetic_rrms", worst)
    # Independently check ideal attention and its much tighter LSE gate too.
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)


@pytest.mark.parametrize("n", [4096, 8192, 16384])
@pytest.mark.parametrize("capacity", [2112, 2176])
def test_large_query_all_rows_single_tail(n, capacity):
    """Analytic oracle for every head/element, independent of chunked kernel calls."""
    torch.manual_seed(81771 + n + capacity)
    q = torch.randn((n, 64, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((257, 512), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    rows = torch.arange(n, device=q.device)
    slots = (rows * 37 + 11) % len(kv)
    positions = 2051 + rows % (capacity - 2051)
    positions[-1] = capacity - 1
    ids = torch.full((n, capacity), -1, device=q.device, dtype=torch.int32)
    ids[rows, positions] = slots.to(torch.int32)
    out, lse = sparse_mla_fp8_fwd(
        q, kv, ids, 1 / 16, q_scale=0.5, kv_scale=1.25, return_lse=True
    )
    # Batch only the independent oracle to bound FP64 temporary storage.
    for start in range(0, n, 32):
        end = min(start + 32, n)
        values = kv.view(torch.uint8)[slots[start:end]].view(kv.dtype).double() * 1.25
        expected = values[:, None, :].expand(-1, 64, -1).to(torch.bfloat16)
        encoded = (q[start:end].float() / 0.5).to(torch.float8_e4m3fn).double() * 0.5
        expected_lse = (encoded * values[:, None, :]).sum(-1) / 16
        torch.testing.assert_close(out[start:end], expected, atol=0, rtol=0)
        torch.testing.assert_close(
            lse[start:end].double(), expected_lse, atol=5e-4, rtol=1e-5
        )


def test_arithmetic_gate_rejects_small_error():
    q, kv, ids = arithmetic_inputs(8, 2176, seed=91737)
    expected, expected_lse = arithmetic_reference(q, kv, ids)
    out, lse = sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
    assert_arithmetic_reference(out, lse, expected, expected_lse)
    with pytest.raises(AssertionError):
        assert_arithmetic_reference(out.float() * 1.02, lse, expected, expected_lse)


def test_query_limit():
    q = torch.empty((16385, 64, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.empty((1, 512), device=q.device, dtype=torch.float8_e4m3fn)
    ids = torch.empty((len(q), 0), device=q.device, dtype=torch.int32)
    for call in [
        lambda: allocate_sparse_mla_fp8_workspace(q),
        lambda: sparse_mla_fp8_fwd(q, kv, ids, 1 / 16),
        # Bypass the Python checks to cover the public custom-op boundary.
        lambda: torch.ops.aiter.sparse_mla_fp8_fwd_out(
            q,
            kv,
            ids,
            torch.empty_like(q),
            None,
            torch.empty(0, device=q.device, dtype=torch.bfloat16),
            torch.empty(0, device=q.device, dtype=torch.float32),
            1 / 16,
            1.0,
            1.0,
        ),
    ]:
        with pytest.raises((ValueError, RuntimeError), match="16384"):
            call()


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

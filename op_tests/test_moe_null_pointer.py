# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional launch operands must retain the Tensor-backed null pointer ABI."""

import importlib
import importlib.util
from pathlib import Path

import flydsl.expr as fx
import pytest
import torch

GM = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
MK = importlib.import_module("aiter.ops.flydsl.moe_kernels")
SHIM = importlib.import_module("aiter.ops.flydsl.kernels.tensor_shim")
FM = importlib.import_module("aiter.fused_moe")
NEWPTR = SHIM.ptr_arg
OLDPTR = NEWPTR


def tensor_null_reference(t, dtype=None):
    # The pre-patch ABI for an omitted operand: a zero-element Tensor whose
    # data_ptr is null. Device is irrelevant to that pointer representation.
    # This is an ABI/correctness oracle, not a performance baseline.
    if t is None:
        t = torch.empty(0, dtype=torch.uint8)
    return NEWPTR(t, dtype)


def select(arm):
    for mod in (GM, MK, SHIM):
        mod.ptr_arg = tensor_null_reference if arm == "baseline" else NEWPTR


@pytest.fixture(autouse=True)
def restore_pointer_aliases():
    before = {mod: mod.ptr_arg for mod in (GM, MK, SHIM)}
    yield
    for mod, pointer in before.items():
        mod.ptr_arg = pointer


def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 grouped MoE")


def setup(m, fmt):
    require_gpu()
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_null_moe_reference", path)
    reference = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    captured = []
    try:
        spec.loader.exec_module(reference)
        public = reference.fused_moe

        def capture(*args, **kw):
            kw["quant_dtype_a"] = FM.dtypes.fp8 if fmt == "a8w4" else FM.dtypes.fp4x2
            captured.append((args, kw))
            return public(*args, **kw)

        reference.fused_moe = capture
        out, ref, _, _ = reference._run_grouped_via_fused_moe(
            experts=8,
            tokens=m,
            topk=2,
            model_dim=256,
            inter_dim=256,
            data_format=fmt,
            activation=FM.ActivationType.Swiglu,
            use_bias=True,
            seed=173,
        )
        assert reference._logits_diff(out, ref) < reference.LOGITS_DIFF_TOL
    finally:
        torch.set_default_device(default)
    return captured[-1]


@pytest.mark.parametrize("dtype", [None, fx.Uint8, fx.Int32, fx.Float32])
def test_null_pointer_abi(dtype):
    empty = torch.empty(0, device="cpu")
    old = OLDPTR(empty, dtype)
    new = NEWPTR(None, dtype)
    assert old.__cache_signature__() == new.__cache_signature__()
    assert old.pointer.value == new.pointer.value is None
    for p in (old, new):
        ctype, fill = p.__c_abi_spec__()[0]
        slot = ctype()
        fill(p, slot)
        assert slot.value is None
    tensor = torch.arange(5, device="cpu")
    assert (
        OLDPTR(tensor, dtype).__cache_signature__()
        == NEWPTR(tensor, dtype).__cache_signature__()
    )
    assert NEWPTR(tensor, dtype).pointer.value == tensor.data_ptr()


@pytest.mark.parametrize("fmt", ["a4w4", "a8w4"])
@pytest.mark.parametrize("m", [1, 16, 257])
def test_full_moe_reference_graph_and_allocations(fmt, m, monkeypatch):
    require_gpu()
    a, kw = setup(m, fmt)
    out = {}
    counts = {}
    for arm in ("baseline", "candidate"):
        select(arm)
        real = torch.empty
        empty_calls = []

        def track(*aa, _real=real, _calls=empty_calls, **kk):
            t = _real(*aa, **kk)
            if not t.numel():
                _calls.append(True)
            return t

        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", track)
            out[arm] = FM.fused_moe(*a, **kw)
            counts[arm] = len(empty_calls)
    assert torch.equal(out["baseline"], out["candidate"])
    assert counts["candidate"] < counts["baseline"], counts
    select("candidate")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            y = FM.fused_moe(*a, **kw)
        for _ in range(3):
            a[0].neg_()
            graph.replay()
            select("baseline")
            ref = FM.fused_moe(*a, **kw)
            assert torch.equal(y, ref)
            select("candidate")
    torch.cuda.current_stream().wait_stream(s)


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("write_rowmap", [False, True])
def test_routes_dynamic_counts_and_graph(truncate, write_rowmap, arm):
    require_gpu()
    select(arm)
    E, T, K, D, tile = 8, 23, 2, 128, 16
    max_m = T * K
    cap = T * K + E * tile
    ids = torch.arange(T * K, device="cuda", dtype=torch.int32).reshape(T, K) % E
    weights = torch.linspace(0.1, 0.9, T * K, device="cuda").reshape(T, K)
    nvalid = torch.tensor([T], device="cuda", dtype=torch.int32)
    nvr = nvalid * K
    expert_mask = torch.ones(E, device="cuda", dtype=torch.int32)
    source = (
        torch.arange(cap * D, device="cuda", dtype=torch.float32).reshape(1, cap, D)
        % 17
    ).to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        routed_weights = (
            torch.empty_like(weights, dtype=torch.bfloat16) if truncate else weights
        )
        counts, rows = MK.flydsl_moe_topids_to_rows(
            ids,
            E,
            max_m,
            expert_mask=expert_mask if truncate else None,
            weight_in=weights if truncate else None,
            gather_w=routed_weights if truncate else None,
            num_valid_routes=nvr if truncate else None,
        )
        rowmap = (
            torch.full((cap,), -1, device="cuda", dtype=torch.int32)
            if write_rowmap
            else None
        )
        start, psum, total = GM.contiguous_psum_remap(
            counts,
            rows,
            E,
            max_m,
            tile,
            num_valid_routes=nvr if truncate else None,
            row_to_token=rowmap,
            topk=K,
        )
        out = torch.full((T, D), -83, device="cuda", dtype=source.dtype)
        out = GM.flydsl_moe_gather_reduce(
            source,
            rows,
            routed_weights,
            out=out,
            num_valid_tokens=nvalid if truncate else None,
        )
        return counts, rows, start, psum, total, rowmap, out

    def check(values):
        count, rows, start, psum, total, rowmap, out = values
        live = int(nvalid.item()) if truncate else T
        expected = torch.full_like(out, -83)
        if live:
            ref_weights = weights.to(torch.bfloat16).float() if truncate else weights
            expected[:live] = (
                (source[0][rows[:live].long()].float() * ref_weights[:live, :, None])
                .sum(1)
                .to(out.dtype)
            )
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.1)
        expected_counts = torch.bincount(
            ids[:live].reshape(-1).long(), minlength=E
        ).int()
        assert torch.equal(count, expected_counts)
        tile_counts = ((expected_counts + tile - 1) // tile) * tile
        expected_starts = (tile_counts.cumsum(0) - tile_counts).int()
        assert torch.equal(start, expected_starts)
        assert torch.equal(psum, expected_starts + expected_counts)
        assert int(total.item()) == max(tile, int(tile_counts.sum().item()))
        if write_rowmap and live:
            assert torch.equal(
                rowmap[rows[:live].long()],
                torch.arange(live, device="cuda")[:, None].expand(live, K).int(),
            )

    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            values = run()
        for live in (T, 7, 0, T):
            nvalid.fill_(live)
            nvr.fill_(live * K)
            ids.add_(1).remainder_(E)
            source.neg_()
            graph.replay()
            check(values)
    torch.cuda.current_stream().wait_stream(stream)


def test_fake_pointer_preserves_metadata():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        tensor = torch.empty((3, 7))
        for dtype in (None, fx.Uint8, fx.Int32, fx.Float32):
            a, b = NEWPTR(tensor, dtype), NEWPTR(None, dtype)
            assert a.__cache_signature__() == b.__cache_signature__()
            assert a.pointer.value == b.pointer.value is None


def test_ep_placeholder_and_live_abi():
    for params in (
        None,
        {
            "masked_m": torch.ones(4, dtype=torch.int32),
            "psum": torch.arange(4, dtype=torch.int32),
            "gather_w": torch.ones((3, 2), dtype=torch.bfloat16),
            "tis": torch.zeros(6, dtype=torch.int32),
            "ep_rowmap": torch.zeros((6, 2), dtype=torch.int32),
            "experts": 4,
            "tile_m": 16,
            "topk": 2,
            "max_tok": 3,
            "slot_stride": 32,
        },
    ):
        select("baseline")
        old = MK._routeks_ep_psum_args("cpu", params)
        select("candidate")
        new = MK._routeks_ep_psum_args("cpu", params)
        assert old[5:] == new[5:]
        for a, b in zip(old[:5], new[:5]):
            assert a.__cache_signature__() == b.__cache_signature__()
            assert a.pointer.value == b.pointer.value

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unused quantizer operand must not require a separate Tensor allocation."""

import ast
import importlib
import inspect
import textwrap

import pytest
import torch

pytest.importorskip("flydsl")
mk = importlib.import_module("aiter.ops.flydsl.moe_kernels")
new = mk.flydsl_moe_fused_quant_preshuffle


def _previous_wrapper():
    """Replace only the new placeholder alias with the original allocation.

    This is a same-kernel regression oracle, not an independent quantization
    reference. The full public MoE comparison uses the canonical numeric oracle.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(new)))
    replaced = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "masked_m"
            and isinstance(node.value, ast.Name)
            and node.value.id == "out_payload"
        ):
            node.value = ast.parse(
                "torch.empty(max(E, 1), dtype=torch.int32, device=device)",
                mode="eval",
            ).body
            replaced += 1
    assert replaced == 1
    ast.fix_missing_locations(tree)
    namespace = dict(vars(mk))
    # The tree comes only from this installed repository function, never user text.
    exec(compile(tree, "<previous-quant-placeholder>", "exec"), namespace)  # noqa: S102
    return namespace[new.__name__]


old = _previous_wrapper()


@pytest.fixture(autouse=True)
def _gpu_contract(request):
    if request.node.name.startswith("test_host_"):
        return
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 quantizer paths")


@pytest.mark.parametrize("masked", [False, True])
def test_host_placeholder_alias_and_live_mask(masked, monkeypatch):
    from aiter.ops.flydsl.kernels import kernels_common

    x = torch.zeros(64, 256, dtype=torch.bfloat16)
    payload = torch.empty(2, 32, 128, dtype=torch.uint8)
    scale = torch.empty(2, 32, 8, dtype=torch.uint8)
    live = torch.tensor([16, 32], dtype=torch.int32) if masked else None
    marker = object()
    seen = []
    monkeypatch.setattr(kernels_common, "get_warp_size", lambda: 32)
    monkeypatch.setattr(mk, "_get_compiled_fused_quant_preshuffle", lambda **kw: marker)
    monkeypatch.setattr(mk, "_dispatch_compiled", lambda *args: seen.append(args))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())

    def unexpected_empty(*args, **kwargs):
        pytest.fail("provided outputs must not allocate an unused mask")

    monkeypatch.setattr(torch, "empty", unexpected_empty)
    got = new(x, 2, 32, wmma_rep=1, masked_m=live, out_payload=payload, out_scale=scale)
    assert got[0] is payload and got[1] is scale
    assert len(seen) == 1 and seen[0][0] is marker
    mask_ptr = seen[0][4]
    assert mask_ptr.pointer.value == (live.data_ptr() if masked else payload.data_ptr())


@pytest.mark.parametrize("mode", ["fp4", "fp8"])
@pytest.mark.parametrize("fd,rep", [(256, 1), (2048, 4), (6144, 8)])
@pytest.mark.parametrize(
    "kind", ["grouped", "masked", "route", "remap", "prequant", "multidest", "compact"]
)
def test_all_quant_paths(mode, fd, rep, kind, monkeypatch):
    device = "cuda"
    E = 2
    T = 64 if kind in ("compact", "multidest") else 16
    M = max(rep * 16 * 2, T)
    topk = 2
    torch.manual_seed(713)
    assert M % (rep * 16) == 0
    source_rows = E * M if kind in ("grouped", "masked") else T
    hidden = torch.randn(source_rows, fd, device=device, dtype=torch.bfloat16) * 0.2
    Pb = fd if mode == "fp8" else fd // 2
    Ws = fd // 32
    ids = torch.arange(T * topk, device=device, dtype=torch.int32).reshape(T, topk)
    # Give every route a unique slot in its expert; remap from larger masked stride.
    experts = ids % 2
    slots = ids // 2
    route_stride = M * 2 if kind == "remap" else M
    route = (experts * route_stride + slots).int()
    starts = torch.tensor([0, M], device=device, dtype=torch.int32)
    masked = torch.tensor([M // 2, M], device=device, dtype=torch.int32)
    seen = []
    for getter in (
        "_get_compiled_token_multidest_compact_quant",
        "_get_compiled_token_multidest_quant",
    ):
        original = getattr(mk, getter)

        def tracked(*a, _getter=getter, _original=original, **kw):
            seen.append(_getter)
            return _original(*a, **kw)

        monkeypatch.setattr(mk, getter, tracked)
    kwargs = {"wmma_rep": rep, "quant_mode": mode}
    if kind == "masked":
        kwargs["masked_m"] = masked
    if kind not in ("grouped", "masked"):
        kwargs.update(topids_to_rows=route, source_topk=topk)
        if kind == "remap":
            kwargs.update(row_starts=starts, route_max_m=route_stride)
        if kind in ("route", "remap", "prequant"):
            monkeypatch.setenv("AITER_FLYDSL_TOKEN_MULTIDEST_QUANT", "0")
        else:
            monkeypatch.setenv("AITER_FLYDSL_TOKEN_MULTIDEST_QUANT", "1")
    if kind == "prequant":
        # Use actual base quantizer to create matching packed bytes/e8m0 scale rows.
        # A rep1 scale layout is still interleaved; explicitly invert its torch layout.
        q, scale = old(hidden, 1, T, wmma_rep=1, quant_mode=mode)
        hidden = q.view(T, Pb).contiguous()
        raw = (
            scale.reshape(1, T // 16, Ws // 4, 1, 16, 4)
            .permute(0, 1, 3, 4, 2, 5)
            .contiguous()
            .reshape(T, Ws)
        )
        kwargs["prequantized_scale"] = raw
    backing = []
    for _ in (0, 1):
        p = torch.full((E * M * Pb + 32,), 173, device=device, dtype=torch.uint8)
        if kind == "compact":
            s = torch.full((T * Ws + 32,), 197, device=device, dtype=torch.uint8)
            scale = s[16:-16].reshape(T, Ws)
            r2t = torch.full((E * M,), -1, device=device, dtype=torch.int32)
        else:
            s = torch.full((E * M * Ws + 32,), 197, device=device, dtype=torch.uint8)
            scale = s[16:-16].reshape(E, M // rep, Ws * rep)
            r2t = None
        backing.append((p, s, p[16:-16].reshape(E, M, Pb), scale, r2t))
    routes = [route.clone(), route.clone()]

    def invoke(i):
        _payload_guard, _scale_guard, payload, sc, r2t = backing[i]
        kw = dict(kwargs, out_payload=payload, out_scale=sc)
        if kind not in ("grouped", "masked"):
            routes[i].copy_(route)
            kw["topids_to_rows"] = routes[i]
        if kind == "compact":
            kw["row_to_token"] = r2t
        return (old if i == 0 else new)(hidden, E, M, **kw)

    def check():
        for i in (0, 1):
            p, s, _, _, _ = backing[i]
            assert bool((p[:16] == 173).all()) and bool((p[-16:] == 173).all())
            assert bool((s[:16] == 197).all()) and bool((s[-16:] == 197).all())
        assert torch.equal(backing[0][0], backing[1][0]), (
            mode,
            fd,
            rep,
            kind,
            "payload",
        )
        assert torch.equal(backing[0][1], backing[1][1]), (
            mode,
            fd,
            rep,
            kind,
            "scales",
        )
        if kind == "compact":
            assert torch.equal(backing[0][4], backing[1][4])
        assert torch.equal(routes[0], routes[1])

    invoke(0)
    invoke(1)
    check()
    if kind in ("compact", "multidest"):
        expected_getter = (
            "_get_compiled_token_multidest_compact_quant"
            if kind == "compact"
            else "_get_compiled_token_multidest_quant"
        )
        # The copied legacy function has its own globals; the live candidate uses mk.
        assert seen and set(seen) == {expected_getter}, (kind, seen)

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        invoke(1)
    torch.cuda.current_stream().wait_stream(st)
    for step in range(3):
        if kind != "prequant":
            hidden.neg_()
        else:
            hidden.bitwise_xor_(255)
        if kind == "masked":
            masked.fill_(0 if step == 0 else M)
        # Sentinel reset is outside capture; padding must stay unwritten each replay.
        for p, s, _, _, r2t in backing:
            p[16:-16].fill_(173)
            s[16:-16].fill_(197)
            if r2t is not None:
                r2t.fill_(-1)
        g.replay()
        invoke(0)
        check()


@pytest.mark.parametrize("mode", ["fp4", "fp8"])
@pytest.mark.parametrize("remapping", ["ep", "plain"])
def test_ep_fused_equals_explicit_remap(mode, remapping):
    T, E, fd, rep, tile = 16, 2, 2048, 4, 64
    M = 128
    masked_stride = 256
    capacity = E * M
    hidden = torch.randn(T, fd, device="cuda", dtype=torch.bfloat16) * 0.2
    base = torch.arange(T * 2, device="cuda", dtype=torch.int32)
    expert = base % 2
    slot = base // 2
    route = expert * masked_stride + slot
    weights = torch.full((T * 2,), 0.5, device="cuda", dtype=torch.bfloat16)
    tis = torch.arange(T, device="cuda", dtype=torch.int32)
    live = torch.tensor([T * 2], device="cuda", dtype=torch.int32)
    counts = torch.full((E,), T, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, tile], device="cuda", dtype=torch.int32)
    pb = fd // 2 if mode == "fp4" else fd
    ws = fd // 32
    state = []
    for _ in range(2):
        state.append(
            {
                "routes": route.clone(),
                "payload": torch.full(
                    (E, M, pb), 173, device="cuda", dtype=torch.uint8
                ),
                "scale": torch.full(
                    (E, M // rep, ws * rep), 197, device="cuda", dtype=torch.uint8
                ),
                "psum": torch.full((E,), -9, device="cuda", dtype=torch.int32),
                "epmap": torch.full(
                    (capacity, 2), -1, device="cuda", dtype=torch.int32
                ),
            }
        )

    def call(i):
        v = state[i]
        v["routes"].copy_(route)
        v["payload"].fill_(173)
        v["scale"].fill_(197)
        v["epmap"].fill_(-1)
        extra = (
            {"row_starts": starts, "route_max_m": masked_stride}
            if remapping == "plain"
            else {
                "ep_psum_params": {
                    "experts": E,
                    "route_max_m": masked_stride,
                    "masked_m": counts,
                    "psum": v["psum"],
                    "gather_w": weights,
                    "tis": tis,
                    "ep_rowmap": v["epmap"],
                    "tile_m": tile,
                    "topk": 2,
                    "max_tok": T,
                    "slot_stride": T * 2,
                }
            }
        )
        (old if i == 0 else new)(
            hidden,
            E,
            M,
            wmma_rep=rep,
            quant_mode=mode,
            topids_to_rows=v["routes"],
            source_topk=2,
            num_valid_routes=live,
            out_payload=v["payload"],
            out_scale=v["scale"],
            **extra
        )

    call(0)
    call(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call(1)
    torch.cuda.current_stream().wait_stream(stream)
    for n in (T * 2, 8, 0, T * 2):
        live.fill_(n)
        counts.fill_(n // 2)
        starts[1] = tile if n else 0
        hidden.neg_()
        graph.replay()
        call(0)
        for key in ("routes", "payload", "scale", "epmap", "psum"):
            assert torch.equal(state[0][key], state[1][key]), (mode, remapping, n, key)
        valid = route[:n] // masked_stride * tile + route[:n] % masked_stride
        assert torch.equal(state[1]["routes"][:n], valid)
        if remapping == "ep":
            assert torch.equal(state[1]["psum"], starts + counts)
            assert torch.equal(
                state[1]["epmap"][valid.long(), 0],
                torch.arange(n, device="cuda", dtype=torch.int32),
            )


@pytest.mark.parametrize("prequant", [False, True])
@pytest.mark.parametrize("fd", [1024, 2048])
def test_apreshuffled_paths(prequant, fd):
    T, E, M, rep = 64, 2, 128, 4
    h = torch.randn(T, fd, device="cuda", dtype=torch.bfloat16) * 0.2
    orig_r = torch.arange(T * 2, device="cuda", dtype=torch.int32)
    rows = (orig_r % 2 * M + orig_r // 2).reshape(T, 2)
    inverse = torch.full((E * M,), -1, device="cuda", dtype=torch.int32)
    inverse[rows.reshape(-1).long()] = torch.arange(
        T, device="cuda", dtype=torch.int32
    ).repeat_interleave(2)
    if prequant:
        q, s = old(h, 1, T, wmma_rep=1, quant_mode="fp4")
        h = q.reshape(T, fd // 2)
        raw = (
            s.reshape(1, T // 16, fd // 128, 1, 16, 4)
            .permute(0, 1, 3, 4, 2, 5)
            .contiguous()
            .reshape(T, fd // 32)
        )
    else:
        raw = None
    outputs = []
    for _ in range(2):
        outputs.append(
            (
                torch.full((E, M, fd // 2), 173, device="cuda", dtype=torch.uint8),
                torch.full(
                    (E, M // rep, fd // 32 * rep), 197, device="cuda", dtype=torch.uint8
                ),
            )
        )

    def call(i):
        p, s = outputs[i]
        p.fill_(173)
        s.fill_(197)
        (old if i == 0 else new)(
            h,
            E,
            M,
            wmma_rep=rep,
            quant_mode="fp4",
            topids_to_rows=rows,
            source_topk=2,
            row_to_token=inverse,
            prequantized_scale=raw,
            payload_preshuffled=True,
            out_payload=p,
            out_scale=s,
        )

    call(0)
    call(1)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        call(1)
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(3):
        if prequant:
            h.bitwise_xor_(255)
        else:
            h.neg_()
        g.replay()
        call(0)
        assert all(torch.equal(a, b) for a, b in zip(*outputs))


@pytest.mark.parametrize("fmt", ["a4w4", "a8w4"])
@pytest.mark.parametrize("tokens", [1, 64])
def test_public_moe_allocation_and_reference(fmt, tokens, monkeypatch):
    import importlib.util
    from pathlib import Path

    fm = importlib.import_module("aiter.fused_moe")
    path = Path(__file__).resolve().parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_placeholder_moe_reference", path)
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    captured = []
    try:
        spec.loader.exec_module(ref)
        public = ref.fused_moe

        def capture(*args, **kwargs):
            kwargs["quant_dtype_a"] = (
                fm.dtypes.fp8 if fmt == "a8w4" else fm.dtypes.fp4x2
            )
            captured.append((args, kwargs))
            return public(*args, **kwargs)

        ref.fused_moe = capture
        output, reference, _, _ = ref._run_grouped_via_fused_moe(
            experts=8,
            tokens=tokens,
            topk=2,
            model_dim=2048,
            inter_dim=256,
            data_format=fmt,
            activation=fm.ActivationType.Swiglu,
            use_bias=True,
            seed=721,
        )
        assert ref._logits_diff(output, reference) < ref.LOGITS_DIFF_TOL
    finally:
        torch.set_default_device(default)
    args, kwargs = captured[-1]
    values, counts = {}, {}
    for label, implementation in (("old", old), ("new", new)):
        allocated = []
        original_empty = torch.empty

        def empty(*a, _record=allocated, _empty=original_empty, **kw):
            value = _empty(*a, **kw)
            _record.append((tuple(value.shape), value.dtype))
            return value

        with monkeypatch.context() as patch:
            patch.setattr(mk, "flydsl_moe_fused_quant_preshuffle", implementation)
            patch.setattr(torch, "empty", empty)
            values[label] = fm.fused_moe(*args, **kwargs)
        counts[label] = allocated
    assert torch.equal(values["old"], values["new"])
    assert counts["old"].count(((1,), torch.int32)) == (
        counts["new"].count(((1,), torch.int32)) + 1
    )
    assert len(counts["old"]) == len(counts["new"]) + 1

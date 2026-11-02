# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch policy tests, requiring no launches or mocked GPU outputs."""

import importlib

import pytest

g = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
q = importlib.import_module("aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter")
m = importlib.import_module("aiter.ops.flydsl.moe_kernels")


@pytest.mark.parametrize("split", [1, 2, 4, 8, 14])
@pytest.mark.parametrize("override", [None, "0", "64", "-1", "10000", "bad", ""])
def test_policy_override_and_auto(monkeypatch, split, override):
    if override is None:
        monkeypatch.delenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", raising=False)
    else:
        monkeypatch.setenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", override)
    monkeypatch.setattr(q, "fused_quant_preshuffle_supported", lambda *a: True)
    monkeypatch.setattr(q, "token_multidest_ksplit", lambda *a: split)
    monkeypatch.setattr(m, "token_multidest_eligible", lambda *a: True)
    expected = override != "10000" and (override in ("0", "64", "-1") or split == 1)
    assert g._use_fused_quant_preshuffle(6144, 8, "fp4", 2517, 5) == expected


@pytest.mark.parametrize(
    "minimum,rows,geometry,eligible",
    [(1024, 1023, True, True), (1024, 2517, False, True), (1024, 2517, True, False)],
)
def test_declined_call_does_not_probe_split(
    monkeypatch, minimum, rows, geometry, eligible
):
    monkeypatch.setenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", str(minimum))
    monkeypatch.setattr(q, "fused_quant_preshuffle_supported", lambda *a: geometry)
    monkeypatch.setattr(m, "token_multidest_eligible", lambda *a: eligible)
    monkeypatch.setattr(
        q,
        "token_multidest_ksplit",
        lambda *a: pytest.fail("declined call reached split"),
    )
    assert not g._use_fused_quant_preshuffle(6144, 8, "fp4", rows, 5)


@pytest.mark.parametrize("dim", [2048, 4096, 6144, 7168])
@pytest.mark.parametrize("rep", [1, 4, 8])
@pytest.mark.parametrize("fmt", ["fp4", "fp8"])
@pytest.mark.parametrize("cus", [64, 128, 256])
def test_real_geometry_boundary(monkeypatch, dim, rep, fmt, cus):
    from aiter.jit.utils import chip_info

    monkeypatch.delenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", raising=False)
    monkeypatch.setattr(q, "get_rocm_arch", lambda: "gfx1250")
    monkeypatch.setattr(q, "get_warp_size", lambda: 32)
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cus)
    for rows in (
        64,
        512,
        1023,
        1024,
        2047,
        2048,
        2517,
        4096,
        cus * 32 - 8,
        cus * 32 - 7,
        cus * 32,
        16384,
    ):
        old = (
            rows >= g._compact_scale_min_tokens(rep)
            and m.token_multidest_eligible(rows, 5)
            and q.fused_quant_preshuffle_supported(dim, rep, fmt)
        )
        want = old and (
            fmt != "fp4" or q.token_multidest_ksplit(dim, rep, fmt, rows) == 1
        )
        assert g._use_fused_quant_preshuffle(dim, rep, fmt, rows, 5) == want


@pytest.mark.parametrize("topk", [0, 1, 9, 16])
def test_outside_multidest_stays_out(monkeypatch, topk):
    monkeypatch.setenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", "0")
    monkeypatch.setattr(
        q, "token_multidest_ksplit", lambda *a: pytest.fail("ineligible topk")
    )
    assert not g._use_fused_quant_preshuffle(6144, 8, "fp4", 16384, topk)


def test_fp8_keeps_policy_without_ksplit_query(monkeypatch):
    monkeypatch.delenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", raising=False)
    monkeypatch.setattr(q, "fused_quant_preshuffle_supported", lambda *a: True)
    monkeypatch.setattr(m, "token_multidest_eligible", lambda *a: True)
    monkeypatch.setattr(
        q,
        "token_multidest_ksplit",
        lambda *a: pytest.fail("FP8 should not query K-split"),
    )
    assert g._use_fused_quant_preshuffle(6144, 8, "fp8", 2517, 5)


@pytest.mark.parametrize("fmt", ["a4w4", "a8w4"])
@pytest.mark.parametrize("rows", [1024, 2517, 8192])
def test_real_policy_paths_reference_and_graph(fmt, rows, monkeypatch):
    import importlib.util
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 quantization policy")
    fm = importlib.import_module("aiter.fused_moe")
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_compact_policy_reference", path)
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    policy = g._use_fused_quant_preshuffle
    calls = []
    rebuild = g.flydsl_moe_scatter_preshuffle_scale

    def track_rebuild(*args, **kwargs):
        calls.append(True)
        return rebuild(*args, **kwargs)

    # Exact previous automatic policy, without replacing any kernels.
    def previous(dim, rep, mode, tokens, topk):
        return (
            tokens >= g._compact_scale_min_tokens(rep)
            and m.token_multidest_eligible(tokens, topk)
            and q.fused_quant_preshuffle_supported(dim, rep, mode)
        )

    try:
        spec.loader.exec_module(ref)
        capture = []
        public = ref.fused_moe

        def save_inputs(*args, **kwargs):
            kwargs["quant_dtype_a"] = (
                fm.dtypes.fp8 if fmt == "a8w4" else fm.dtypes.fp4x2
            )
            capture.append((args, kwargs))
            return public(*args, **kwargs)

        ref.fused_moe = save_inputs
        monkeypatch.delenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", raising=False)
        monkeypatch.setattr(g, "flydsl_moe_scatter_preshuffle_scale", track_rebuild)
        with monkeypatch.context() as old:
            old.setattr(g, "_use_fused_quant_preshuffle", previous)
            output, reference, _, _ = ref._run_grouped_via_fused_moe(
                experts=8,
                tokens=rows,
                topk=5,
                model_dim=2048,
                inter_dim=256,
                data_format=fmt,
                activation=fm.ActivationType.Swiglu,
                use_bias=True,
                seed=617,
            )
        assert calls, "baseline must reach compact scale reconstruction"
        assert ref._logits_diff(output, reference) < ref.LOGITS_DIFF_TOL
        args, kwargs = capture[-1]
        calls.clear()
        current = fm.fused_moe(*args, **kwargs)
        assert len(calls) == int(fmt == "a8w4" or rows >= 8192)
        assert torch.equal(output, current)
        # Explicit override retains existing tuning semantics, even for K-split.
        with monkeypatch.context() as force:
            force.setenv("AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS", "0")
            calls.clear()
            forced = fm.fused_moe(*args, **kwargs)
            assert calls
            assert torch.equal(forced, current)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                replayed = fm.fused_moe(*args, **kwargs)
            for _ in range(2):
                args[0].neg_()
                graph.replay()
                with monkeypatch.context() as old:
                    old.setattr(g, "_use_fused_quant_preshuffle", previous)
                    expected = fm.fused_moe(*args, **kwargs)
                assert torch.equal(replayed, expected)
        torch.cuda.current_stream().wait_stream(stream)
        assert g._use_fused_quant_preshuffle is policy
    finally:
        torch.set_default_device(default)

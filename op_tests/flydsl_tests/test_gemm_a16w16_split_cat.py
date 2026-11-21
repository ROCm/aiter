# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Focused tests for the gfx1250 BF16 GEMM split-cat epilogue.

CPU/meta tests cover the public contract and compile-cache specialization:
    pytest op_tests/flydsl_tests/test_gemm_a16w16_split_cat.py -q

The correctness cases require gfx1250 and compare bitwise against the unchanged
GEMM followed by split/cat:
    pytest op_tests/flydsl_tests/test_gemm_a16w16_split_cat.py -q -k bitwise
"""

import importlib.util
import inspect
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER_PATH = _REPO_ROOT / "aiter/ops/flydsl/gemm_a16w16_gfx1250.py"

HEADS = 128
K_DIM = 512
K_NOPE = 128
V_DIM = 128
ROPE_DIM = 64
N_DIM = HEADS * (K_NOPE + V_DIM)

_CONFIG = {
    "tile_m": 128,
    "tile_n": 256,
    "tile_k": 64,
    "m_warp": 2,
    "n_warp": 4,
    "num_buffers": 2,
}


def _meta(shape, stride=None):
    if stride is None:
        return torch.empty(shape, dtype=torch.bfloat16, device="meta")
    return torch.empty_strided(shape, stride, dtype=torch.bfloat16, device="meta")


def _metadata_case(m=31):
    x = _meta((m, K_DIM))
    w = _meta((N_DIM, K_DIM))
    k_head_stride = K_NOPE + ROPE_DIM + 8
    v_head_stride = V_DIM + 8
    k_out = _meta(
        (m, HEADS, K_NOPE + ROPE_DIM),
        (HEADS * k_head_stride + 16, k_head_stride, 1),
    )
    v_out = _meta(
        (m, HEADS, V_DIM),
        (HEADS * v_head_stride + 16, v_head_stride, 1),
    )
    k_rope = _meta((m, 1, ROPE_DIM), (ROPE_DIM + 8, ROPE_DIM, 1))
    return x, w, k_out, v_out, k_rope


@pytest.fixture
def split_cat_module(monkeypatch):
    """Load the wrapper without importing AITER's GPU-initializing package root."""
    aiter_stub = types.ModuleType("aiter")
    aiter_stub.__path__ = []
    utility_stub = types.ModuleType("aiter.utility")
    utility_stub.__path__ = []
    graph_alloc_stub = types.ModuleType("aiter.utility.graph_alloc")
    graph_alloc_stub.capture_scoped_cache = lambda fn: fn
    graph_alloc_stub.persistent_alloc = lambda _device: nullcontext()
    monkeypatch.setitem(sys.modules, "aiter", aiter_stub)
    monkeypatch.setitem(sys.modules, "aiter.utility", utility_stub)
    monkeypatch.setitem(sys.modules, "aiter.utility.graph_alloc", graph_alloc_stub)
    spec = importlib.util.spec_from_file_location(
        "_test_gemm_a16w16_gfx1250", _WRAPPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_cat_api_is_opt_in(split_cat_module):
    parameters = inspect.signature(split_cat_module.gemm_a16w16).parameters
    for name in ("k_out", "v_out", "k_rope", "split_sizes"):
        assert name in parameters
        assert parameters[name].default is None

    x, w, *_ = _metadata_case()
    assert (
        split_cat_module._validate_split_cat(
            x,
            w,
            torch.bfloat16,
            None,
            None,
            None,
            None,
            None,
        )
        is None
    )


def test_split_cat_metadata_accepts_head_and_row_pitch(split_cat_module):
    x, w, k_out, v_out, k_rope = _metadata_case()
    spec = split_cat_module._validate_split_cat(
        x,
        w,
        torch.bfloat16,
        None,
        k_out,
        v_out,
        k_rope,
        (K_NOPE, V_DIM),
    )
    assert spec == {
        "k_split_size": K_NOPE,
        "v_split_size": V_DIM,
        "rope_size": ROPE_DIM,
        "heads": HEADS,
        "k_row_stride": k_out.stride(0),
        "k_head_stride": k_out.stride(1),
        "v_row_stride": v_out.stride(0),
        "v_head_stride": v_out.stride(1),
        "rope_row_stride": k_rope.stride(0),
    }


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda case: (*case[:2], case[2], None, case[4]), "provided together"),
        (
            lambda case: (
                *case[:2],
                _meta((31, HEADS, K_NOPE + ROPE_DIM - 8)),
                case[3],
                case[4],
            ),
            "k_out width",
        ),
        (
            lambda case: (
                *case[:2],
                _meta(
                    (31, HEADS, K_NOPE + ROPE_DIM),
                    (
                        HEADS * (K_NOPE + ROPE_DIM),
                        K_NOPE + ROPE_DIM,
                        2,
                    ),
                ),
                case[3],
                case[4],
            ),
            "last-dimension contiguous",
        ),
    ],
)
def test_split_cat_metadata_rejects_invalid_contract(split_cat_module, mutate, match):
    x, w, k_out, v_out, k_rope = mutate(_metadata_case())
    with pytest.raises(ValueError, match=match):
        split_cat_module._validate_split_cat(
            x,
            w,
            torch.bfloat16,
            None,
            k_out,
            v_out,
            k_rope,
            (K_NOPE, V_DIM),
        )


def test_default_and_split_cat_compile_configs_are_separate(
    split_cat_module, monkeypatch
):
    seen = []

    def compile_stub(**kwargs):
        seen.append(kwargs)
        return kwargs

    monkeypatch.setattr(split_cat_module, "_compile_gemm_a16w16", compile_stub)
    split_cat_module._cached_launcher.cache_clear()
    split_cat_module._cached_split_cat_launcher.cache_clear()

    base = tuple(range(len(split_cat_module._CFG_KEYS)))
    assert split_cat_module._cached_launcher(*base) == dict(
        zip(split_cat_module._CFG_KEYS, base)
    )
    assert not {"split_cat", "k_split_size", "v_split_size", "rope_size"} & set(
        seen[-1]
    )

    fused = base + (True, K_NOPE, V_DIM, ROPE_DIM)
    assert split_cat_module._cached_split_cat_launcher(*fused) == dict(
        zip(split_cat_module._SPLIT_CAT_CFG_KEYS, fused)
    )
    assert seen[-1]["split_cat"] is True


def test_wrapper_routes_only_opt_in_calls_to_split_cat(split_cat_module, monkeypatch):
    launches = []
    legacy_launcher = object()
    fused_launcher = object()
    stream = types.SimpleNamespace(cuda_stream=17)
    monkeypatch.setattr(split_cat_module, "_lazy_import", lambda: None)
    monkeypatch.setattr(split_cat_module, "_p", lambda tensor: tensor)
    monkeypatch.setattr(
        split_cat_module, "_split_k_counters", lambda _device, _stream: object()
    )
    monkeypatch.setattr(
        split_cat_module, "_cached_launcher", lambda *_cfg: legacy_launcher
    )
    monkeypatch.setattr(
        split_cat_module,
        "_cached_split_cat_launcher",
        lambda *_cfg: fused_launcher,
    )
    monkeypatch.setattr(
        split_cat_module, "_run_compiled", lambda *args: launches.append(args)
    )
    monkeypatch.setattr(
        split_cat_module,
        "_fx",
        types.SimpleNamespace(Stream=lambda handle: ("stream", handle)),
    )
    monkeypatch.setattr(
        split_cat_module.torch.cuda,
        "current_stream",
        lambda device: stream,
    )

    x, w, k_out, v_out, k_rope = _metadata_case()
    y = _meta((x.shape[0], w.shape[0]))
    assert split_cat_module.gemm_a16w16(x, w, dtype=torch.bfloat16, y=y) is y
    assert launches[-1][0] is legacy_launcher
    assert len(launches[-1]) == 12

    result = split_cat_module.gemm_a16w16(
        x,
        w,
        dtype=torch.bfloat16,
        k_out=k_out,
        v_out=v_out,
        k_rope=k_rope,
        split_sizes=(K_NOPE, V_DIM),
    )
    assert result[0] is k_out and result[1] is v_out
    assert launches[-1][0] is fused_launcher
    assert len(launches[-1]) == 20


def _is_gfx1250():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    return arch.startswith("gfx1250")


_GPU_ONLY = pytest.mark.skipif(not _is_gfx1250(), reason="gfx1250 GPU required")


def _pitched_output(m, heads, width, head_pad, row_pad):
    head_stride = width + head_pad
    row_stride = heads * head_stride + row_pad
    return torch.empty_strided(
        (m, heads, width),
        (row_stride, head_stride, 1),
        dtype=torch.bfloat16,
        device="cuda",
    )


@_GPU_ONLY
@pytest.mark.parametrize("m", [1, 31, 32, 128, 1024, 9424, 16384])
def test_gemm_a16w16_split_cat_bitwise(m):
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

    torch.manual_seed(0)
    x = torch.randn((m, K_DIM), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N_DIM, K_DIM), dtype=torch.bfloat16, device="cuda")
    k_rope = torch.randn((m, 1, ROPE_DIM), dtype=torch.bfloat16, device="cuda")
    if m == 31:
        k_out = _pitched_output(m, HEADS, K_NOPE + ROPE_DIM, 8, 16)
        v_out = _pitched_output(m, HEADS, V_DIM, 8, 16)
    else:
        k_out = torch.empty(
            (m, HEADS, K_NOPE + ROPE_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_out = torch.empty((m, HEADS, V_DIM), dtype=torch.bfloat16, device="cuda")

    baseline = gemm_a16w16(x, w, dtype=torch.bfloat16, **_CONFIG).view(
        m, HEADS, K_NOPE + V_DIM
    )
    expected_k = torch.cat(
        (baseline[..., :K_NOPE], k_rope.expand(-1, HEADS, -1)), dim=-1
    )
    expected_v = baseline[..., K_NOPE:]

    result = gemm_a16w16(
        x,
        w,
        dtype=torch.bfloat16,
        k_out=k_out,
        v_out=v_out,
        k_rope=k_rope,
        split_sizes=(K_NOPE, V_DIM),
        **_CONFIG,
    )
    assert result[0] is k_out and result[1] is v_out
    assert torch.equal(k_out, expected_k)
    assert torch.equal(v_out, expected_v)

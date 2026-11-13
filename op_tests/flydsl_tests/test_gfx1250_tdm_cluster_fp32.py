# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP32 cluster split-K parity, precision, lifetime, and legacy CSV aliases."""

import importlib
import inspect
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.kernels.gemm_a16w16_tdm_gfx1250 import (
    build_gemm_gfx1250,
    validate_gemm_gfx1250_config,
)

wrapper = importlib.import_module("aiter.ops.flydsl.gemm_a16w16_tdm")
gemm = wrapper.gemm_a16w16_tdm
REF_CFG = {
    "block_m": 16,
    "block_n": 16,
    "block_k": 256,
    "stages": 5,
    "m_waves": 1,
    "n_waves": 1,
    "split_k": 5,
}


@pytest.fixture(autouse=True)
def gfx1250_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("requires gfx1250")
    torch.set_num_threads(1)


def guarded(rows, cols, dtype):
    backing = torch.full(
        (rows + 2, (cols + 7) // 8 * 8 + 16),
        17,
        device="cuda",
        dtype=dtype,
    )
    return backing[1 : rows + 1, 8 : cols + 8], backing


def check_guards(backing, cols):
    assert torch.all(backing[0] == 17) and torch.all(backing[-1] == 17)
    assert torch.all(backing[:, :8] == 17) and torch.all(backing[:, cols + 8 :] == 17)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout,interleave", [("nt", False), ("nn", False), ("nt", True)]
)
@pytest.mark.parametrize("split_k", range(2, 10))
def test_all_splits_layouts_tails_and_graph(dtype, layout, interleave, split_k):
    torch.manual_seed(6161)
    m, n, k = 133, 139, split_k * 128
    a, ag = guarded(m, k, dtype)
    weight, bg = guarded(n, k, dtype) if layout == "nt" else guarded(k, n, dtype)
    b = weight.t() if layout == "nt" else weight
    out, og = guarded(m, n, dtype)
    # Binary fractions give exact FP32 dot products and avoid a tolerant test
    # masking missing splits. Both signs and cancellation are exercised.
    a.copy_(torch.randint(-4, 5, a.shape, device="cuda").float() / 16)
    b.copy_(torch.randint(-4, 5, b.shape, device="cuda").float() / 16)
    bias = torch.arange(n, device="cuda").remainder(5).to(dtype) / 8
    cfg = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "split_k": split_k,
        "layout": layout,
        "interleave": interleave,
    }
    gemm(a, b, out, bias, **cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(a, b, out, bias, **cfg)
    for epoch in range(6):
        if epoch:
            a.neg_()
            bias.neg_()
        reference = (a.float() @ b.float()).to(dtype)
        reference = (reference.float() + bias.float()).to(dtype)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
    for storage, cols in ((ag, k), (bg, k if layout == "nt" else n), (og, n)):
        check_guards(storage, cols)
    graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias_on", [False, True])
def test_ref_case_preserves_fp32_partials(dtype, bias_on, monkeypatch):
    a = torch.zeros(6, 5120, device="cuda", dtype=dtype)
    b = torch.zeros(32, 5120, device="cuda", dtype=dtype).t()
    c = torch.empty(6, 32, device="cuda", dtype=dtype)
    factor = 1 / 16 if dtype == torch.bfloat16 else 1 / 64
    for part in range(5):
        a[:, part * 1024] = b[part * 1024, :] = 1
        a[:, part * 1024 + 1] = b[part * 1024 + 1, :] = factor
    bias = torch.full((32,), -0.5, device="cuda", dtype=dtype) if bias_on else None
    expected = torch.full_like(c, 5 * (1 + factor**2))
    if bias_on:
        expected = (expected.float() + bias.float()).to(dtype)
    allocations = []
    original = torch.empty

    def capture(*args, **kwargs):
        t = original(*args, **kwargs)
        allocations.append(t)
        return t

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", capture)
        gemm(a, b, c, bias, **REF_CFG)
    torch.cuda.synchronize()
    assert len(allocations) == 1
    ws = allocations[0]
    assert ws.dtype == torch.float32 and ws.numel() == 2 * 5 * 16 * 16
    # Only six valid rows: padded partial elements are zero, the others must
    # retain the contribution below BF16/FP16 rounding precision.
    nonzero = ws[ws != 0]
    assert nonzero.numel() == 5 * 6 * 32
    torch.testing.assert_close(
        nonzero, torch.full_like(nonzero, 1 + factor**2), atol=0, rtol=0
    )
    torch.testing.assert_close(c, expected, atol=0, rtol=0)


@pytest.mark.parametrize("policy,canonical", [("tdma", "tdm"), ("tdmia", "tdmi")])
def test_old_csv_names_are_deterministic_fp32_aliases(policy, canonical):
    torch.manual_seed(731)
    a = torch.randn(128, 768, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(128, 768, device="cuda", dtype=a.dtype) / 8
    cfg = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "split_k": 3,
    }
    old_mode = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        reference = flydsl_hgemm(a, b, policy=canonical, **cfg)
        for _ in range(8):
            actual = flydsl_hgemm(a, b, policy=policy, **cfg)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(old_mode)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_same_origin_capture_concurrent_replay_and_explicit_stream(dtype):
    slots = []
    for idx in range(2):
        a = torch.full((6, 5120), (idx + 1) / 16, device="cuda", dtype=dtype)
        b = torch.full((32, 5120), 1 / 32, device="cuda", dtype=dtype).t()
        out = torch.empty(6, 32, device="cuda", dtype=dtype)
        gemm(a, b, out, **REF_CFG)
        slots.append((a, b, out))
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graphs = []
    for a, b, out in slots:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            intermediate = torch.empty(4096, device="cuda")
            intermediate.fill_(999)
            del intermediate
            gemm(a, b, out, stream=capture_stream, **REF_CFG)
        graphs.append(graph)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(100):
        for st, gr in zip(streams, graphs):
            with torch.cuda.stream(st):
                gr.replay()
    torch.cuda.synchronize()
    for idx, (_, _, out) in enumerate(slots):
        torch.testing.assert_close(
            out, torch.full_like(out, (idx + 1) * 10), atol=0, rtol=0
        )
    for _ in range(20):
        for st, slot in zip(streams, slots):
            gemm(*slot, stream=st, **REF_CFG)
    torch.cuda.synchronize()
    for idx, (_, _, out) in enumerate(slots):
        torch.testing.assert_close(
            out, torch.full_like(out, (idx + 1) * 10), atol=0, rtol=0
        )
    for graph in graphs:
        graph.reset()


def test_split1_no_workspace_or_counter_allocation(monkeypatch):
    a = torch.ones(6, 5120, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(32, 5120, device="cuda", dtype=a.dtype).t()
    c = torch.empty(6, 32, device="cuda", dtype=a.dtype)
    cfg = dict(REF_CFG, split_k=1)
    gemm(a, b, c, **cfg)
    torch.cuda.synchronize()

    def no_alloc(*args, **kwargs):
        raise AssertionError("split1 must not allocate scratch")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", no_alloc)
        patch.setattr(torch, "zeros", no_alloc)
        gemm(a, b, c, **cfg)
    torch.cuda.synchronize()


def test_no_atomic_implementation_and_device_cache_key():
    assert "split_k_atomic" not in inspect.signature(build_gemm_gfx1250).parameters
    assert "device_id" in inspect.signature(build_gemm_gfx1250).parameters
    assert (
        "split_k_atomic"
        not in inspect.signature(validate_gemm_gfx1250_config).parameters
    )
    cfg = {
        "dtype_id": 2,
        "block_m": 16,
        "block_n": 16,
        "block_k": 256,
        "stages": 5,
        "m_waves": 1,
        "n_waves": 1,
        "group_m": 0,
        "has_bias": False,
        "b_k_contiguous": True,
        "mma_m": 16,
        "mma_n": 16,
        "mma_k": 32,
        "split_k": 5,
    }
    a = build_gemm_gfx1250(**cfg, device_id=0)
    assert a is build_gemm_gfx1250(**cfg, device_id=0)
    # Factory identity check only; no dispatch to a nonexistent second GPU.
    assert a is not build_gemm_gfx1250(**cfg, device_id=1)


def test_cluster_isa_no_atomic_instructions(monkeypatch, tmp_path):
    shim = importlib.import_module("aiter.ops.flydsl.kernels.tensor_shim")
    obj = shutil.which("llvm-objdump")
    if obj is None:
        tools = list(
            Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core").rglob(
                "llvm-objdump"
            )
        )
        if not tools:
            pytest.skip("llvm-objdump required")
        obj = str(tools[0])
    captured = []
    original = shim._run_compiled

    def capture(exe, *args, **kwargs):
        ret = original(exe, *args, **kwargs)
        captured.append(exe._cf)
        return ret

    monkeypatch.setattr(shim, "_run_compiled", capture)
    a = torch.ones(6, 5120, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(32, 5120, device="cuda", dtype=a.dtype).t()
    gemm(a, b, **REF_CFG)
    torch.cuda.synchronize()
    ir = captured[-1]._keepalive._ir_text
    assert "_fp32sk" in ir
    raw = re.search(r'bin = "((?:\\.|[^"\\])*)"', ir)[1]
    binary = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == "\\":
            if re.fullmatch("[0-9a-fA-F]{2}", raw[i + 1 : i + 3]):
                binary.append(int(raw[i + 1 : i + 3], 16))
                i += 3
            else:
                binary.extend(raw[i + 1].encode())
                i += 2
        else:
            binary.extend(raw[i].encode())
            i += 1
    elf = tmp_path / "fp32_cluster.co"
    elf.write_bytes(binary)
    asm = subprocess.check_output([obj, "-d", "--mcpu=gfx1250", str(elf)], text=True)
    ops = [
        line.split()[0] for line in asm.splitlines() if "//" in line and line.strip()
    ]
    assert not any("atomic" in op for op in ops)
    assert "s_barrier_signal" in ops and "s_barrier_wait" in ops
    assert any(op in ("buffer_store_b128", "buffer_store_dwordx4") for op in ops)
    assert any(op in ("buffer_load_b128", "buffer_load_dwordx4") for op in ops)

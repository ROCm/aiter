# SPDX-License-Identifier: MIT
import copy
import importlib

import pytest
import torch

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1151", reason="gfx1151 FP4 GEMM regression tests"
)


def _inputs(m, n, k, strided=False):
    torch.manual_seed(42)
    x = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    xs = torch.randint(124, 128, (m, k // 32), dtype=torch.uint8, device="cuda")
    ws = torch.randint(124, 128, (n, k // 32), dtype=torch.uint8, device="cuda")
    if strided:
        x, w, xs, ws = [t.t().contiguous().t() for t in (x, w, xs, ws)]
    return x, w, xs, ws


def _reference(x, w, xs, ws):
    # Independent E2M1/E8M0 decoder; no Triton or aiter quantizer involved.
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=x.device,
        dtype=torch.float32,
    )

    def decode(q, scales):
        codes = torch.stack((q & 15, q >> 4), dim=-1).flatten(1).long()
        scale = torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
        return values[codes] * scale

    return decode(x, xs) @ decode(w, ws).T


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1, 32),
        (3, 37, 96),
        (16, 128, 512),
        (17, 65, 160),
        (128, 256, 1024),
        (129, 129, 288),
        (512, 256, 512),
        (1024, 128, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_gfx1151_fp4_gemm(m, n, k, dtype, strided):
    args = _inputs(m, n, k, strided)
    expected = _reference(*args)
    y = torch.empty((m, n), dtype=dtype, device="cuda")
    actual = gemm_afp4wfp4(*args, dtype=dtype, y=y)
    assert actual.data_ptr() == y.data_ptr()
    assert torch.isfinite(actual).all()
    # Compare FP32 accumulation directly. FP16/BF16 additionally round stores.
    rtol = {torch.float32: 1e-4, torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=rtol)


@pytest.mark.parametrize("skip_reduce", [False, True])
def test_gfx1151_fp4_splitk(skip_reduce):
    args = _inputs(17, 129, 1024)
    config, _ = _get_config(17, 129, 512)
    config["NUM_KSPLIT"] = 4
    actual = gemm_afp4wfp4(
        *args, dtype=torch.float32, config=config, skip_reduce=skip_reduce
    )
    if skip_reduce:
        assert actual.ndim == 3
        actual = actual.sum(0)
    torch.testing.assert_close(actual, _reference(*args), atol=0.002, rtol=1e-4)


@pytest.mark.parametrize("m", [1, 16, 17, 128, 129, 512])
def test_gfx1151_fp4_compiled_shared_memory(m, monkeypatch):
    module = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_afp4wfp4")
    kernel = module._triton_gemm_afp4wfp4_kernel
    run = kernel.run
    metadata = []

    def record(*args, **kwargs):
        compiled = run(*args, **kwargs)
        if compiled is not None:
            metadata.append(compiled.metadata)
        return compiled

    monkeypatch.setattr(kernel, "run", record)
    args = _inputs(m, 256, 1024)
    gemm_afp4wfp4(*args)
    torch.cuda.synchronize()
    assert metadata, "The test must observe an actual compiled/loaded GPU kernel."
    assert all(meta.shared <= 65536 for meta in metadata)


def test_gfx1151_fp4_graph_and_config_reuse():
    args = _inputs(16, 128, 512)
    config, _ = _get_config(16, 128, 256)
    before = copy.deepcopy(config)
    for _ in range(3):
        expected = gemm_afp4wfp4(*args, config=config)
    assert config == before
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = gemm_afp4wfp4(*args, config=config)
    args[0].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(result) == 0
    assert torch.count_nonzero(expected) > 0


@pytest.mark.parametrize("scale", [100, 112, 127, 140, 150])
def test_gfx1151_fp4_scale_range(scale):
    args = _inputs(3, 37, 96)
    args[2].fill_(scale)
    args[3].fill_(127)
    expected = _reference(*args)
    actual = gemm_afp4wfp4(*args, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-4)


def test_gfx1151_fp4_zero_operand():
    args = _inputs(17, 65, 160)
    args[0].zero_()
    actual = gemm_afp4wfp4(*args, dtype=torch.float32)
    assert torch.count_nonzero(actual) == 0

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.triton.gemm.basic import gemm_a8w8_blockscale as triton_mod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a8w8 blockscale fallback tests require a CUDA/HIP device",
)

BLOCK = 128
N, K = 2560, 1280
CK_CONFIG = {"libtype": "ck", "splitK": 0, "kernelName": "tuned_kernel"}
SPLIT_K_CONFIG = {"libtype": "ck", "splitK": 2, "kernelName": "tuned_kernel"}
MIN_M_BY_GFX = gemm_mod._BLOCKSCALE_TRITON_FALLBACK_MIN_M
NO_THRESHOLD_GFX = sorted(
    gemm_mod._BLOCKSCALE_HIP_PREBUILT_ARCHES - MIN_M_BY_GFX.keys()
)

MIN_M = MIN_M_BY_GFX.get(gemm_mod.get_gfx() if torch.cuda.is_available() else None)
needs_threshold = pytest.mark.skipif(MIN_M is None, reason="no threshold for this arch")


def dispatch(
    monkeypatch, m, config=None, gfx=None, dtype=dtypes.bf16, operand_dtype=dtypes.fp8
):
    reached = []

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, **kwargs):
        reached.append("ck_tuned" if kwargs.get("kernelName") else "ck")
        return Y

    def fake_backend(name):
        def run(XQ, WQ, x_scale, w_scale, Y, *args, **kwargs):
            reached.append(name)
            return Y

        return run

    def fake_triton(xq, wq, x_scale, w_scale, dtype=dtypes.bf16, **kwargs):
        reached.append("triton")
        return torch.empty(xq.shape[0], wq.shape[0], dtype=dtype, device=xq.device)

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_ck", fake_ck)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_cktile", fake_backend("cktile"))
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_flydsl", fake_backend("flydsl"))
    monkeypatch.setattr(triton_mod, "gemm_a8w8_blockscale", fake_triton)
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *a, **kw: config)
    if gfx is not None:
        monkeypatch.setattr(gemm_mod, "get_gfx", lambda: gfx)

    xq = torch.zeros((m, K), device="cuda", dtype=operand_dtype)
    wq = torch.zeros((N, K), device="cuda", dtype=operand_dtype)
    x_scale = torch.ones((m, K // BLOCK), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((N // BLOCK, K // BLOCK), device="cuda", dtype=torch.float32)
    gemm_mod.gemm_a8w8_blockscale(xq, wq, x_scale, w_scale, dtype)

    assert len(reached) == 1, f"expected one kernel call, got {reached}"
    return reached[0]


@needs_threshold
@pytest.mark.parametrize("delta", [0, 1, 4096])
def test_gemm_a8w8_blockscale_uses_triton_for_untuned_at_or_above_threshold(
    monkeypatch, delta
):
    assert dispatch(monkeypatch, MIN_M + delta) == "triton"


@needs_threshold
@pytest.mark.parametrize("delta", [1, 128])
def test_gemm_a8w8_blockscale_uses_ck_for_untuned_below_threshold(monkeypatch, delta):
    assert dispatch(monkeypatch, MIN_M - delta) == "ck"


@needs_threshold
@pytest.mark.parametrize("delta", [0, 4096])
def test_gemm_a8w8_blockscale_uses_ck_for_tuned_above_threshold(monkeypatch, delta):
    assert dispatch(monkeypatch, MIN_M + delta, config=CK_CONFIG) == "ck_tuned"


@needs_threshold
@pytest.mark.parametrize("delta, untuned", [(-1, "ck"), (0, "triton")])
def test_gemm_a8w8_blockscale_skips_split_k_row_for_fp16(monkeypatch, delta, untuned):
    m = MIN_M + delta
    assert dispatch(monkeypatch, m, config=SPLIT_K_CONFIG, dtype=dtypes.fp16) == untuned
    assert dispatch(monkeypatch, m, config=SPLIT_K_CONFIG) == "ck_tuned"
    assert dispatch(monkeypatch, m, config=CK_CONFIG, dtype=dtypes.fp16) == "ck_tuned"


@needs_threshold
@pytest.mark.parametrize("libtype", ["ck", "cktile", "flydsl"])
@pytest.mark.parametrize("operand_dtype", [torch.float8_e5m2, torch.int8])
def test_gemm_a8w8_blockscale_skips_tuned_row_for_non_e4m3_operands(
    monkeypatch, libtype, operand_dtype
):
    config = {**CK_CONFIG, "libtype": libtype}
    reached = dispatch(monkeypatch, MIN_M, config=config, operand_dtype=operand_dtype)
    assert reached == "triton"
    tuned = "ck_tuned" if libtype == "ck" else libtype
    assert dispatch(monkeypatch, MIN_M, config=config) == tuned
    assert (
        dispatch(monkeypatch, MIN_M, config=config, operand_dtype=torch.uint8) == tuned
    )


@pytest.mark.parametrize("gfx", NO_THRESHOLD_GFX)
def test_gemm_a8w8_blockscale_uses_ck_on_arch_without_threshold(monkeypatch, gfx):
    assert dispatch(monkeypatch, 65536, gfx=gfx) == "ck"


@needs_threshold
def test_gemm_a8w8_blockscale_triton_fallback_matches_ck():
    gen = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape):
        return torch.rand(shape, generator=gen, device="cuda", dtype=torch.float32)

    xq = (rand(MIN_M, K) / 4).to(dtypes.fp8)
    wq = (rand(N, K) / 4).to(dtypes.fp8)
    x_scale = rand(MIN_M, K // BLOCK)
    w_scale = rand(N // BLOCK, K // BLOCK)

    routed = gemm_mod.gemm_a8w8_blockscale(xq, wq, x_scale, w_scale, dtypes.bf16)
    expected = torch.empty(MIN_M, N, dtype=dtypes.bf16, device="cuda")
    gemm_mod.gemm_a8w8_blockscale_ck(xq, wq, x_scale, w_scale, expected)

    torch.testing.assert_close(routed, expected, rtol=2e-2, atol=2e-2)

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_mod.get_gfx() not in ("gfx942", "gfx950"),
    reason="the blockscale bpreshuffle asm kernels ship for gfx942 and gfx950",
)

N, K = 2560, 6144
PAD = 192
SENTINEL = 7.0
ASM_KERNEL = "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E"


def make_inputs(m, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(m, K, device="cuda") * 0.5).to(dtypes.fp8)
    w = (torch.randn(N, K, device="cuda") * 0.5).to(dtypes.fp8)
    # bytes column-major [K/128, M], shaped (M, K/128), as the per-1x128 quant emits
    x_scale = (
        (torch.rand(m, K // 128, device="cuda") + 0.5).t().contiguous().view(m, -1)
    )
    w_scale = torch.rand(N // 128, K // 128, device="cuda") + 0.5
    return x, shuffle_weight(w, layout=(16, 16)), x_scale, w_scale


def run_backend(backend, x, w, x_scale, w_scale, out):
    if backend == "ck":
        aiter.gemm_a8w8_blockscale_bpreshuffle_ck(x, w, x_scale, w_scale, out)
    elif backend == "cktile":
        aiter.gemm_a8w8_blockscale_bpreshuffle_cktile(x, w, x_scale, w_scale, out)
    else:
        split_k = int(backend.removeprefix("asm_sk"))
        aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
            x, w, out, x_scale, w_scale, splitK=split_k, kernelName=ASM_KERNEL
        )


@pytest.mark.parametrize("m", [32, 120])
@pytest.mark.parametrize("backend", ["ck", "cktile", "asm_sk1", "asm_sk4"])
def test_row_strided_out_matches_contiguous(backend, m):
    x, w, x_scale, w_scale = make_inputs(m)
    ref = torch.empty(m, N, dtype=dtypes.bf16, device="cuda")
    run_backend(backend, x, w, x_scale, w_scale, ref)

    wide = torch.full((m, N + PAD), SENTINEL, dtype=dtypes.bf16, device="cuda")
    out = wide[:, :N]
    run_backend(backend, x, w, x_scale, w_scale, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=0.02)
    assert bool((wide[:, N:] == SENTINEL).all()), "wrote outside the out view"


@pytest.mark.parametrize("backend", ["ck", "asm_sk1", "asm_sk4"])
def test_single_row_out_ignores_size_one_stride(backend):
    x, w, x_scale, w_scale = make_inputs(1)
    ref = torch.empty(1, N, dtype=dtypes.bf16, device="cuda")
    run_backend(backend, x, w, x_scale, w_scale, ref)

    out = torch.empty(N, 1, dtype=dtypes.bf16, device="cuda").t()
    assert out.stride() == (1, 1)
    run_backend(backend, x, w, x_scale, w_scale, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=0.02)


@pytest.mark.parametrize("backend", ["ck", "asm_sk1", "asm_sk4"])
def test_overlapping_out_rows_raise(backend):
    m = 32
    x, w, x_scale, w_scale = make_inputs(m)
    storage = torch.empty(m * N, dtype=dtypes.bf16, device="cuda")
    out = storage.as_strided((m, N), (N - 128, 1))
    with pytest.raises(RuntimeError):
        run_backend(backend, x, w, x_scale, w_scale, out)


def test_split_k_out_needs_four_byte_alignment():
    m = 32
    x, w, x_scale, w_scale = make_inputs(m)
    wide = torch.empty(m, N + 2, dtype=dtypes.bf16, device="cuda")
    with pytest.raises(RuntimeError):
        run_backend("asm_sk4", x, w, x_scale, w_scale, wide[:, 1 : N + 1])


def test_fp16_fallback_warns_once_per_shape(monkeypatch):
    asm_row = {"libtype": "asm", "splitK": 4, "kernelName": ASM_KERNEL}
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *args, **kwargs: asm_row)
    messages = []
    monkeypatch.setattr(
        gemm_mod.logger, "warning", lambda msg, *a, **kw: messages.append(msg)
    )
    gemm_mod._warn_blockscale_asm_row_not_bf16.cache_clear()

    for m in (16, 32, 48):
        x, w, x_scale, w_scale = make_inputs(m)
        gemm_mod.gemm_a8w8_blockscale_bpreshuffle(x, w, x_scale, w_scale, dtypes.fp16)

    assert len([msg for msg in messages if "asm" in msg]) == 1


def test_fp16_output_on_asm_row_uses_ck(monkeypatch):
    m = 32
    x, w, x_scale, w_scale = make_inputs(m)
    asm_row = {"libtype": "asm", "splitK": 4, "kernelName": ASM_KERNEL}
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *args, **kwargs: asm_row)
    asm_calls = []
    real_asm = gemm_mod.gemm_a8w8_blockscale_bpreshuffle_asm

    def spy_asm(*args, **kwargs):
        asm_calls.append(kwargs.get("splitK"))
        return real_asm(*args, **kwargs)

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_bpreshuffle_asm", spy_asm)

    ref = gemm_mod.gemm_a8w8_blockscale_bpreshuffle(x, w, x_scale, w_scale, dtypes.bf16)
    assert asm_calls == [4], "bf16 output should keep using the tuned asm row"

    y = gemm_mod.gemm_a8w8_blockscale_bpreshuffle(x, w, x_scale, w_scale, dtypes.fp16)
    torch.cuda.synchronize()
    assert y.dtype == dtypes.fp16
    assert asm_calls == [4], "fp16 output must not reach the bf16-only asm kernel"
    torch.testing.assert_close(y.float(), ref.float(), atol=0.5, rtol=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

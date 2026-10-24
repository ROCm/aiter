# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a8w8 bpreshuffle pad-K tests require a CUDA/HIP device",
)


def test_preshuffle_flat_buffers_stay_below_4gib():
    from aiter.ops.flydsl.gemm_kernels import (
        _check_preshuffle_flat_buffer_capacity,
    )

    # The output is just under 4 GiB; masking the padded M rows keeps this legal.
    _check_preshuffle_flat_buffer_capacity(8161, 262192, 512, 1, 1, 2)

    with pytest.raises(RuntimeError, match="output buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(8191, 262192, 512, 1, 1, 2)
    with pytest.raises(RuntimeError, match="A buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(65536, 16, 65536, 1, 1, 2)
    with pytest.raises(RuntimeError, match="B buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(16, 65536, 65536, 1, 1, 2)


@pytest.mark.parametrize("m, backend", [(65536, "flydsl"), (65537, "ck")])
def test_flydsl_bpreshuffle_row_falls_back_to_ck_above_m_max(monkeypatch, m, backend):
    from aiter.ops.flydsl import gemm_kernels

    # Kimi-K3's 6144x7168 M=8192 row, which padded-M lookup returns for every
    # M > 8192. Meta tensors stand in for the 470 MB activation.
    row = {
        "libtype": "flydsl",
        "splitK": 0,
        "kernelName": "flydsl_bpreshuflle_128x256x128_F8_F8_B16_1x2x4x2_default",
    }
    n, k = 6144, 7168
    xq = torch.empty((m, k), device="meta", dtype=dtypes.fp8)
    wq = torch.empty((n, k), device="meta", dtype=dtypes.fp8)
    x_scale = torch.empty((m, 1), device="meta", dtype=torch.float32)
    w_scale = torch.empty((n, 1), device="meta", dtype=torch.float32)
    out = torch.empty((m, n), device="meta", dtype=torch.bfloat16)
    seen = []

    def fake_flydsl(*args, **kwargs):
        seen.append("flydsl")

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK=0):
        seen.append("ck")
        return Y

    # gfx1250 routes every kernel name to its WMMA backend before the M-limit check.
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_kernels, "flydsl_preshuffle_gemm_a8", fake_flydsl)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    got = gemm_mod.gemm_a8w8_bpreshuffle_flydsl(xq, wq, x_scale, w_scale, out, row)

    assert gemm_kernels.PRESHUFFLE_M_MAX == 65536
    assert seen == [backend]
    assert got is out


def test_two_wave_vgpr_estimate_uses_128_threads():
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        _estimate_max_wpe,
    )

    # Both tiles hold eight accumulator VGPRs per thread: 32x32 / 128 threads
    # for the 2-wave path and 32x64 / 256 threads for the standard 4-wave path.
    assert _estimate_max_wpe(32, 32, 512, total_vgpr=12) == 1
    assert _estimate_max_wpe(32, 64, 512, total_vgpr=12) == 1


def test_shuffle_weight_pad_k_to_pads_last_dim():
    weight = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)

    shuffled = shuffle_weight(weight, layout=(16, 16), pad_k_to=128)

    assert shuffled.shape == (16, 128)
    assert shuffled.is_shuffled
    assert shuffled.aiter_original_k == 96
    assert shuffled.aiter_padded_k == 128


def test_gemm_a8w8_bpreshuffle_pads_activation_for_ck(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "ck", "splitK": 0}

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_pads_activation_for_cktile(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "cktile", "splitK": 0}

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_falls_back_to_padded_k_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))
        if k == 128:
            return {"libtype": "cktile", "splitK": 0}
        return None

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_pads_activation_for_untuned_cktile(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    def fake_ck(*args, **kwargs):
        raise AssertionError("padded-K untuned fallback should use CKTile")

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_pads_activation_for_flydsl(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "flydsl", "splitK": 0, "kernelName": "fake"}

    def fake_flydsl(XQ, WQ, x_scale, w_scale, Y, config):
        seen["x_shape"] = tuple(XQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_flydsl", fake_flydsl)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_rejects_short_weight_k():
    xq = torch.zeros((2, 128), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="WQ K >= XQ K"):
        gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)


def _padded_k_backend(m, n, k, w_k):
    """(libtype, split-K?) the dispatcher picks for a K-padded weight, or None."""
    for kk in (k, w_k):
        config = gemm_mod.get_GEMM_config_with_quant_type(
            m, n, kk, dtypes.fp8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
        )
        if config is not None:
            return config["libtype"], int(config["splitK"]) > 0
    return None


def run_torch(xq, x_scale, wq, w_scale, dtype=dtypes.bf16):
    # Reference only: fp32 dequant GEMM over the logical K. Not timed, not in the table.
    out = (xq.to(dtypes.fp32) * x_scale) @ (wq.to(dtypes.fp32) * w_scale).t()
    return out.to(dtype)


@benchmark()
def test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtype):
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtypes.bf16)
    w = torch.randn((n, k), dtype=dtypes.bf16)
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    ref = run_torch(xq, x_scale, wq, w_scale, dtype)
    # As in the model: the weight is preshuffled once with K padded to w_k and
    # the logical-K activation goes straight into the dispatcher.
    wq_padded = shuffle_weight(wq, layout=(16, 16), pad_k_to=w_k)
    assert wq_padded.shape == (n, w_k)

    candidates = {
        "a8w8_bpreshuffle": lambda: gemm_mod.gemm_a8w8_bpreshuffle(
            xq, wq_padded, x_scale, w_scale, dtype=dtype
        ),
    }

    backend = _padded_k_backend(m, n, k, w_k)
    # Split-K sums bf16 partials, so small outputs pick up ~1 ulp of the row
    # magnitude; a misread weight is still off by O(|ref|).
    atol = 1.0 if backend is not None and backend[1] else 1e-2

    # Every backend runs at the padded K: XQ is zero-padded to w_k before launch.
    flops = 2 * m * n * w_k
    nbytes = (m * w_k + n * w_k) * xq.element_size() + m * n * ref.element_size()

    ret = {
        "gfx": get_gfx(),
        "backend": (
            "untuned"
            if backend is None
            else backend[0] + ("-splitk" if backend[1] else "")
        ),
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=atol,
            msg=f"{name}: M={m} N={n} K={k}->{w_k} ",
            catastrophic_check=True,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# Shape-arg sweep fn, not a pytest test: the parametrized wrapper below calls it.
test_gemm_a8w8_bpreshuffle_pad_k.__test__ = False


# (case id, expected (libtype, split-K?), (m, n, k, w_k)) per (gfx, cu_num) tuned
# table: rows reaching each CK/CKTile/FlyDSL path with a K-padded weight, either
# through the logical-K row or the padded-K fallback lookup, plus untuned shapes.
_UNTUNED_CASES = [
    ("untuned", None, (37, 1280, 448, 512)),
    ("untuned-half", None, (37, 1280, 256, 512)),
]
_PADDED_K_CASES = {
    ("gfx950", 256): _UNTUNED_CASES
    + [
        ("ck-logical-k", ("ck", False), (1, 64, 4096, 4224)),
        ("ck-splitk-logical-k", ("ck", True), (4, 64, 4096, 4224)),
        ("cktile-logical-k", ("cktile", False), (1, 1536, 128, 256)),
        ("ck-padded-k-row", ("ck", False), (32, 384, 7040, 7168)),
        ("cktile-padded-k-row", ("cktile", False), (2048, 384, 7040, 7168)),
        ("flydsl-padded-k-row", ("flydsl", False), (512, 384, 7040, 7168)),
    ],
    # The 304-CU gfx942 table has no fp8 rows, so only the untuned fallback runs.
    ("gfx942", 304): _UNTUNED_CASES,
    ("gfx942", 80): _UNTUNED_CASES
    + [
        ("ck-logical-k", ("ck", False), (1, 64, 7168, 7296)),
        ("cktile-logical-k", ("cktile", False), (1, 2048, 2048, 2176)),
        ("ck-padded-k-row", ("ck", False), (1, 64, 7040, 7168)),
        ("cktile-padded-k-row", ("cktile", False), (1, 2048, 1920, 2048)),
    ],
}

# Cards the CK/CKTile bpreshuffle paths are built for. gfx1250 has no CK path,
# and its untuned FlyDSL fallback is not covered by these shapes.
SUPPORTED_GFX = ["gfx942", "gfx950"]
# checkAllclose default: max fraction of mismatching elements per case.
TOL_ERR_RATIO = 0.05


@pytest.mark.parametrize(
    "device,backend,mnk",
    [(dev, *c[1:]) for dev, cases in _PADDED_K_CASES.items() for c in cases],
    ids=[
        f"{gfx}-cu{cu}-{c[0]}"
        for (gfx, cu), cases in _PADDED_K_CASES.items()
        for c in cases
    ],
)
def test_gemm_a8w8_bpreshuffle_padded_k_matches_reference(device, backend, mnk):
    # Dispatch keys the tuned CSV with (get_gfx(), get_cu_num()), and the
    # untuned fallback is arch-specific too, so a row only holds on the full
    # device its table was picked from, not on a partitioned one.
    if (get_gfx(), get_cu_num()) != device:
        pytest.skip(f"rows are picked from the {device[0]} cu_num={device[1]} table")
    m, n, k, w_k = mnk
    assert (
        _padded_k_backend(m, n, k, w_k) == backend
    ), f"M={m} N={n} K={k}->{w_k} no longer selects {backend}; pick another row"
    ret = test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtypes.bf16)
    assert ret["a8w8_bpreshuffle err"] <= TOL_ERR_RATIO, ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a8w8_bpreshuffle padded K unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="""Output data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            c[2] for c in _PADDED_K_CASES.get((get_gfx(), get_cu_num()), _UNTUNED_CASES)
        ],
        help="""Shape m,n,k,w_k: k is the activation K, w_k the padded weight K.
    e.g.: -s 37,1280,448,512""",
    )
    args = parser.parse_args()

    df = []
    for dtype, (m, n, k, w_k) in itertools.product(args.dtype, args.mnk):
        df.append(test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtype))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "gemm_a8w8_bpreshuffle padded K summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    bad = df[df["a8w8_bpreshuffle err"] > TOL_ERR_RATIO]
    assert bad.empty, f"padded-K mismatch above {TOL_ERR_RATIO}:\n{bad}"


if __name__ == "__main__":
    main()

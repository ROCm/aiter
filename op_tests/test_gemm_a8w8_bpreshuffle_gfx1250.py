# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the gfx1250 (WMMA) a8w8 bpreshuffle GEMM.

On gfx1250, aiter.gemm_a8w8_bpreshuffle dispatches its FlyDSL path to the WMMA
backend (bpreshuffle_gemm_gfx1250); its tuned kernelNames carry the prefix
``flydsl_bpreshuffle_wmma_``. Semantics are the ordinary a8w8 per-token
(x_scale[M]) / per-channel (w_scale[N]) fp8 GEMM, so inputs are quantized exactly
like the standard a8w8 path. Skipped off gfx1250.
"""

import pytest
import torch

import aiter
from aiter.utility import dtypes
from aiter.test_common import checkAllclose
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import wmma_kernel_name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="gfx1250 WMMA a8w8 bpreshuffle requires a gfx1250 device",
)


def _kernel_name(
    tile_m,
    tile_n,
    tile_k,
    num_buffers,
    split_k=1,
    cluster_m=1,
    cluster_n=1,
    m_warp=2,
    n_warp=2,
):
    return wmma_kernel_name(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_buffers=num_buffers,
        split_k=split_k,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        m_warp=m_warp,
        n_warp=n_warp,
    )


def _quant(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_scale = aiter.pertoken_quant(a, quant_dtype=dtypes.fp8)  # [M, 1]
    bq, b_scale = aiter.pertoken_quant(b, quant_dtype=dtypes.fp8)  # [N, 1]
    return aq, bq, a_scale, b_scale


def _ref(aq, bq, a_scale, b_scale, dtype):
    a_f = aq.to(torch.float32) * a_scale.to(torch.float32)
    b_f = bq.to(torch.float32) * b_scale.to(torch.float32)
    return (a_f @ b_f.t()).to(dtype)


def _inject_tuned_config(monkeypatch, name):
    import aiter.ops.gemm_op_a8w8 as gmod

    config = {"libtype": "flydsl", "splitK": 1, "kernelName": name}
    monkeypatch.setattr(gmod, "get_GEMM_config_with_quant_type", lambda *a, **k: config)


def _assert_close(ref, out, *, split_k=1, msg=""):
    rtol = atol = 2e-2 if split_k > 1 else 1e-2
    bound = 0.10 if split_k > 1 else 0.05
    err = checkAllclose(
        ref, out, rtol=rtol, atol=atol, msg=msg, catastrophic_check=True
    )
    assert err <= bound, f"{msg}: {err:.1%} of elements exceed tol (bound {bound:.0%})"


def test_kernel_name_roundtrips():
    """Every catalogue kernelName must decode back to its config."""
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import parse_wmma_kernel_name
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
        kernels_list,
    )

    assert kernels_list, "WMMA catalogue is empty"
    for ki in kernels_list.values():
        cfg = parse_wmma_kernel_name(ki.name)
        assert cfg is not None, f"cannot parse {ki.name}"
        assert (cfg["tile_m"], cfg["tile_n"], cfg["tile_k"]) == (
            ki.tile_m,
            ki.tile_n,
            ki.tile_k,
        )
        assert cfg["num_buffers"] == ki.num_buffers and cfg["split_k"] == ki.split_k
        assert cfg["cluster_m"] == ki.cluster_m and cfg["cluster_n"] == ki.cluster_n
        assert cfg["m_warp"] == ki.m_warp and cfg["n_warp"] == ki.n_warp


@pytest.mark.parametrize(
    "M,N,K",
    [
        # decode (small M; ragged -> kernel OOB-clips, no host pad)
        (1, 4096, 4096),  # M=1 decode
        (2, 1280, 8192),  # M=2 decode, qkv_proj-like
        (4, 8192, 1024),  # M=4 decode, attn_out-like
        # square / balanced
        (256, 256, 256),
        (512, 1024, 512),
        # ragged M (partial last M-tile)
        (17, 64, 512),  # tiny ragged M, min N (= tile_n=64)
        (100, 256, 512),  # ragged M
        (333, 576, 1024),  # ragged M, N=576 (9*tile_n)
        # prefill / production projections
        (128, 1280, 8192),  # qkv_proj
        (1024, 1280, 8192),  # qkv_proj, large M
        (512, 8192, 1024),  # attn_out
        (2048, 8192, 1024),  # attn_out, large M
        # large N
        (64, 7424, 8192),  # hipmm preshuffle (N=7424 = 116*tile_n)
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_bpreshuffle_gfx1250(M, N, K, dtype, monkeypatch):
    _inject_tuned_config(monkeypatch, _kernel_name(128, 64, 128, 2))
    torch.manual_seed(0)
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, dtype)
    bq_prepared = shuffle_weight(bq, layout=(16, 16))
    out = aiter.gemm_a8w8_bpreshuffle(aq, bq_prepared, a_scale, b_scale, dtype=dtype)

    assert out.shape == (M, N)
    assert out.dtype == dtype
    _assert_close(ref, out, msg=f"M={M}, N={N}, K={K}, dtype={dtype}")


@pytest.mark.parametrize("num_buffers", [2, 3, 4])
def test_num_buffers(num_buffers, monkeypatch):
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, num_buffers))
    torch.manual_seed(0)
    M, N, K = 256, 256, 1024  # K/tile_k = 8 >= 4 buffers
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(
        aq, shuffle_weight(bq, layout=(16, 16)), a_scale, b_scale, dtype=torch.bfloat16
    )
    _assert_close(ref, out, msg=f"num_buffers={num_buffers}")


@pytest.mark.parametrize("split_k", [2, 4])
def test_split_k(split_k, monkeypatch):
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=split_k))
    torch.manual_seed(0)
    M, N, K = 256, 256, 1024
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    bq_sh = shuffle_weight(bq, layout=(16, 16))
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(aq, bq_sh, a_scale, b_scale, dtype=torch.bfloat16)
    _assert_close(ref, out, split_k=split_k, msg=f"split_k={split_k}")

    # split-k accumulates partial tiles via bf16 atomics, so it tracks split_k=1
    # closely (cos~1) but not bit-exactly.
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=1))
    out_sk1 = aiter.gemm_a8w8_bpreshuffle(
        aq, bq_sh, a_scale, b_scale, dtype=torch.bfloat16
    )
    _assert_close(out_sk1, out, split_k=split_k, msg=f"split_k={split_k} vs split_k=1")


@pytest.mark.parametrize("split_k", [1, 2, 4])
@pytest.mark.parametrize(
    "M",
    [
        1,  # extreme ragged: 1 of tile_m=128 rows valid (decode)
        17,  # small sub-tile ragged
        100,  # sub-tile ragged
        128,  # aligned control (no remainder)
        257,  # 2 full tiles + 1 ragged row
        300,  # 2 full tiles + 44 ragged rows
        700,  # 5 full tiles + 60 ragged rows
    ],
)
def test_ragged_m_split_k(M, split_k, monkeypatch):
    """Ragged M with split-k exercises the per-lane (row < M) atomic predicate.

    M spans sub-tile, tile-aligned, and multi-tile remainders against tile_m=128.
    """
    torch.manual_seed(0)
    N, K = 256, 1024  # K/(split_k*tile_k) integral; chunk holds >= 2 buffers
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    bq_sh = shuffle_weight(bq, layout=(16, 16))

    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=split_k))
    out = aiter.gemm_a8w8_bpreshuffle(aq, bq_sh, a_scale, b_scale, dtype=torch.bfloat16)
    assert out.shape == (M, N)

    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=1))
    out_sk1 = aiter.gemm_a8w8_bpreshuffle(
        aq, bq_sh, a_scale, b_scale, dtype=torch.bfloat16
    )
    _assert_close(
        out_sk1, out, split_k=split_k, msg=f"ragged M={M} split_k={split_k} vs sk1"
    )


@pytest.mark.parametrize(
    "m_warp,n_warp,tile_m,tile_n",
    [(1, 2, 16, 32), (1, 4, 16, 64), (2, 2, 32, 32), (2, 4, 32, 64)],
)
def test_warp_configs(m_warp, n_warp, tile_m, tile_n, monkeypatch):
    """m_warp/n_warp are tunable; m_warp=1 small-tile_m configs serve decode."""
    _inject_tuned_config(
        monkeypatch,
        _kernel_name(tile_m, tile_n, 256, 2, m_warp=m_warp, n_warp=n_warp),
    )
    torch.manual_seed(0)
    M, N, K = 1, 256, 512
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(
        aq, shuffle_weight(bq, layout=(16, 16)), a_scale, b_scale, dtype=torch.bfloat16
    )
    assert out.shape == (M, N)
    _assert_close(ref, out, msg=f"mw{m_warp} nw{n_warp} t{tile_m}x{tile_n}")


def test_vendored_oob_path(monkeypatch):
    """Force the vendored OOB descriptor builder (the older-flydsl fallback) and
    verify ragged-M correctness, even when the installed flydsl has native OOB.

    Uses a tile config no other test compiles so the cached kernel is built fresh
    through the vendored path while ``_TDM_HAS_OOB`` is patched off.
    """
    import aiter.ops.flydsl.kernels.gemm_fp8fp4_gfx1250 as kmod

    monkeypatch.setattr(kmod, "_TDM_HAS_OOB", False)
    _inject_tuned_config(monkeypatch, _kernel_name(64, 64, 128, 2))
    torch.manual_seed(0)
    M, N, K = 100, 256, 512  # ragged M -> partial last M-tile via vendored desc
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(
        aq, shuffle_weight(bq, layout=(16, 16)), a_scale, b_scale, dtype=torch.bfloat16
    )
    assert out.shape == (M, N)
    _assert_close(ref, out, msg="vendored OOB path")


def test_cluster(monkeypatch):
    """Workgroup cluster (cluster_m/n>1) over an evenly divisible grid."""
    _inject_tuned_config(
        monkeypatch, _kernel_name(128, 128, 128, 2, cluster_m=2, cluster_n=2)
    )
    torch.manual_seed(0)
    M, N, K = 512, 512, 512  # grid (4, 4) divisible by cluster (2, 2)
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(
        aq, shuffle_weight(bq, layout=(16, 16)), a_scale, b_scale, dtype=torch.bfloat16
    )
    _assert_close(ref, out, msg="cluster_m=2 cluster_n=2")


def test_backend_direct_writes_out():
    """The gfx1250 backend writes into the caller's Out tensor."""
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
        run_preshuffle_gemm_a8_gfx1250,
    )

    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ret = run_preshuffle_gemm_a8_gfx1250(
        aq, shuffle_weight(bq, layout=(16, 16)), a_scale, b_scale, out, 128, 128, 128
    )
    assert ret.data_ptr() == out.data_ptr()
    _assert_close(ref, out, msg="backend direct out")


@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("M", [128, 100])
def test_strided_a_input(M, split_k, monkeypatch):
    """A as a row-slice of a wider buffer (stride(0) > K, e.g. a DeepSeek fused
    activation) must run in place -- no contiguous copy -- and match the dense
    result. Exercises the runtime lda path through the vendored TDM descriptor."""
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=split_k))
    torch.manual_seed(0)
    N, K = 256, 1024  # K/(split_k*tile_k) integral; chunk holds >= 2 buffers
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)

    pad = 64  # leading-dim padding -> stride(0) = K + pad
    big = torch.empty(M, K + pad, dtype=aq.dtype, device="cuda")
    big[:, :K].copy_(aq)
    aq_strided = big[:, :K]
    assert aq_strided.stride(0) == K + pad and aq_strided.stride(1) == 1
    assert not aq_strided.is_contiguous()

    out = aiter.gemm_a8w8_bpreshuffle(
        aq_strided,
        shuffle_weight(bq, layout=(16, 16)),
        a_scale,
        b_scale,
        dtype=torch.bfloat16,
    )
    assert out.shape == (M, N)
    _assert_close(
        ref, out, split_k=split_k, msg=f"strided A (M={M}, split_k={split_k})"
    )


@pytest.mark.parametrize("split_k", [1, 2])
def test_strided_c_output(split_k, monkeypatch):
    """Backend writes into a strided (column-sliced) Out without copying, and
    leaves the leading-dim padding gap untouched. Exercises the runtime ldc path
    for both the TDM store (split_k=1) and the atomic-add store (split_k>1)."""
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
        run_preshuffle_gemm_a8_gfx1250,
    )

    torch.manual_seed(0)
    M, N, K = 128, 256, 1024
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)

    pad = 64  # Out is a column-slice of a [M, N+pad] buffer -> stride(0) = N + pad
    big = torch.full((M, N + pad), -1.0, dtype=torch.bfloat16, device="cuda")
    out = big[:, :N]
    assert out.stride(0) == N + pad and not out.is_contiguous()

    ret = run_preshuffle_gemm_a8_gfx1250(
        aq,
        shuffle_weight(bq, layout=(16, 16)),
        a_scale,
        b_scale,
        out,
        128,
        128,
        128,
        split_k=split_k,
    )
    assert ret.data_ptr() == out.data_ptr()  # wrote in place, no copy-back
    _assert_close(ref, out, split_k=split_k, msg=f"strided C (split_k={split_k})")
    assert torch.all(big[:, N:] == -1.0), "kernel wrote into the C padding gap"

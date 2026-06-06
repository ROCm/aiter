# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the gfx1250 (WMMA) a8w8 bpreshuffle GEMM.

aiter.gemm_a8w8_bpreshuffle routes its FlyDSL path to the WMMA backend
(bpreshuffle_gemm_gfx1250) by the kernelName prefix ``flydsl_bpreshuffle_wmma_``.
Semantics are the ordinary a8w8 per-token (x_scale[M]) / per-channel (w_scale[N])
fp8 GEMM, so inputs are quantized exactly like the standard a8w8 path. Skipped off
gfx1250.
"""

import pytest
import torch

import aiter
from aiter.utility import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import wmma_kernel_name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="gfx1250 WMMA a8w8 bpreshuffle requires a gfx1250 device",
)


def _kernel_name(
    tile_m, tile_n, tile_k, num_buffers, split_k=1, cluster_m=1, cluster_n=1
):
    return wmma_kernel_name(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_buffers=num_buffers,
        split_k=split_k,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )


def _metrics(out, ref):
    out_f, ref_f = out.float(), ref.float()
    rel = (out_f - ref_f).abs().sum() / ref_f.abs().sum().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0)
    return rel.item(), cos.item()


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


def test_kernel_name_roundtrips():
    """Every catalogue kernelName must decode back to its config."""
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import parse_wmma_kernel_name
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
        kernels_list,
    )

    assert kernels_list, "WMMA catalogue is empty"
    ki = kernels_list[0]
    cfg = parse_wmma_kernel_name(ki.name)
    assert cfg is not None, f"cannot parse {ki.name}"
    assert (cfg["tile_m"], cfg["tile_n"], cfg["tile_k"]) == (
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
    )
    assert cfg["num_buffers"] == ki.num_buffers and cfg["split_k"] == ki.split_k
    assert cfg["cluster_m"] == ki.cluster_m and cfg["cluster_n"] == ki.cluster_n


@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 256, 256),
        (512, 1024, 512),
        (1, 4096, 4096),  # decode: M=1 (ragged, kernel OOB-clips; no host pad)
        (333, 576, 1024),  # ragged M, N=576 (tile_n=64 divides it)
        (17, 64, 512),  # tiny ragged M, small N
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
    rel, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (M={M},N={N},K={K})"
    assert rel < 0.05, f"rel L1={rel} too high (M={M},N={N},K={K})"


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
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (num_buffers={num_buffers})"


@pytest.mark.parametrize("split_k", [2, 4])
def test_split_k(split_k, monkeypatch):
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=split_k))
    torch.manual_seed(0)
    M, N, K = 256, 256, 1024
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    bq_sh = shuffle_weight(bq, layout=(16, 16))
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(aq, bq_sh, a_scale, b_scale, dtype=torch.bfloat16)
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (split_k={split_k})"

    # split-k must accumulate in fp32
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=1))
    out_sk1 = aiter.gemm_a8w8_bpreshuffle(
        aq, bq_sh, a_scale, b_scale, dtype=torch.bfloat16
    )
    rel, cos = _metrics(out, out_sk1)
    assert rel < 1e-3, f"split_k={split_k} drifts from sk1 (rel L1={rel})"
    assert cos > 0.9999, f"split_k={split_k} drifts from sk1 (cos={cos})"


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("M", [17, 100, 257])
def test_ragged_m_split_k(M, split_k, monkeypatch):
    """Ragged M with split-k exercises the per-lane (row < M) atomic predicate."""
    _inject_tuned_config(monkeypatch, _kernel_name(128, 128, 128, 2, split_k=split_k))
    torch.manual_seed(0)
    N, K = 256, 1024  # K/(split_k*tile_k) integral; chunk holds >= 2 buffers
    aq, bq, a_scale, b_scale = _quant(M, N, K)
    ref = _ref(aq, bq, a_scale, b_scale, torch.bfloat16)
    out = aiter.gemm_a8w8_bpreshuffle(
        aq, shuffle_weight(bq, layout=(16, 16)), a_scale, b_scale, dtype=torch.bfloat16
    )
    assert out.shape == (M, N)
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (ragged M={M}, split_k={split_k})"


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
    rel, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (vendored OOB path)"
    assert rel < 0.05, f"rel L1={rel} too high (vendored OOB path)"


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
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low"


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
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low"

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified correctness + performance test for jagged_dense_bmm backward (jdbba bwd).

Correctness (pytest, Mi=512): dispatch config resolution, cos > 0.999 vs torch-eager
reference on all three grads (dJagged, dDense, dBias), tail-over-read regression,
forced split=2 reduce-path coverage, end-to-end autograd at headline shapes,
multi-device backward (cuda:1 tensors, current device cuda:0), and large-B
(n_groups>=8192) int32 offset boundary at D=512.

Performance (``main()``, Mi=7680): FlyDSL vs upstream Triton on headline deployment
shapes, swept over regime and backward component (``jagged`` / ``dense_bias`` / ``all``).
Split-component timings reuse hoisted output buffers on both sides; Triton
``dense_bias`` still allocates ``d_bias`` internally each iteration (upstream API
has no out-buffer for it), so that component comparison is slightly Triton-heavy
vs FlyDSL.

Run (inside the venv):
    HIP_VISIBLE_DEVICES=0 pytest op_tests/test_jagged_dense_bmm_bwd.py
    HIP_VISIBLE_DEVICES=0 python op_tests/test_jagged_dense_bmm_bwd.py
    HIP_VISIBLE_DEVICES=0 python op_tests/test_jagged_dense_bmm_bwd.py --worker-d 256
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, run_perftest

try:
    import flydsl.compiler as flyc

    from aiter.ops.flydsl import jagged_dense_bmm_bwd_dispatched
    from aiter.ops.flydsl.kernels import jagged_dense_bmm_bwd as _bwd

    _HAS_FLYDSL = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    _HAS_FLYDSL = False
    flyc = None  # type: ignore[assignment]
    jagged_dense_bmm_bwd_dispatched = None  # type: ignore[assignment,misc]
    _bwd = None  # type: ignore[assignment]

torch.set_default_device("cuda")

SUPPORTED_GFX = ("gfx942", "gfx950")
SEED = 1234
_COS_THRESH = 0.999

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_bwd_dense_bias,
        triton_jagged_dense_bmm_add_bwd_jagged,
    )

    _HAS_TRITON = True
except ModuleNotFoundError as _exc:  # pragma: no cover
    _HAS_TRITON = False
    _TRITON_ERR = _exc

# Headline correctness cases per D: (B, Mi, regime). Mi is modest to bound eager-ref
# cost; deployment Mi is exercised via resolve_config (7680) and the perf sweep.
_HEADLINE = [
    (120, 512, "uniform"),
    (120, 512, "genrec"),
    (120, 512, "skew"),
    (1024, 512, "genrec"),
    (1024, 512, "skew"),
]
_REGRESSION = {"B": 32, "Mi": 2048, "regime": "genrec", "seed": 1234, "sparsity": 0.95}
# Large n_groups with minimal Mi to bound memory and eager-ref cost. Group 8192
# hits off_b*D*D = 2^31 in int32 (b_row_off in grad_jagged); Mi=1 keeps L small.
# Run only in _worker(D=512).
_LARGE_B_OVERFLOW = [
    (8193, 1, "uniform"),
]

_PERF_SHAPES = [
    (120, 256, 256),
    (120, 512, 512),
    (1024, 256, 256),
    (1024, 512, 512),
]
_PERF_MI = 7680


def _make_seq_offsets(B, Mi, regime, seed=SEED, device="cuda", sparsity=0.95):
    """Per-group prefix-sum offsets.

    uniform: every group length == Mi.
    skew:    M_i = floor(Mi * U^4), ~20% empty groups, plus one full (Mi) and
             one near-full (0.9*Mi) group.
    genrec:  M_i = Uniform{1, ..., Mi} * sparsity, clamped >= 1 (local recipe;
             similar sparsity intent to HSTU's generate_sparse_seq_len, not identical).
    Mi is the max_seq_len envelope, not the per-group length.
    """
    if regime == "uniform":
        return torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device=device)
    if regime == "genrec":
        g = torch.Generator(device=device).manual_seed(seed)
        lengths = torch.randint(1, Mi + 1, size=(B,), device=device, generator=g)
        if sparsity < 1.0:
            lengths = (lengths.float() * sparsity).clamp(min=1.0).to(torch.int64)
        so = torch.zeros(B + 1, dtype=torch.int32, device=device)
        so[1:] = torch.cumsum(lengths, dim=0).to(torch.int32)
        return so
    g = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(B, generator=g, device=device)
    t = (Mi * (u**4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0  # ~20% empty groups
    t[-1] = Mi  # one full-envelope group
    if B > 1:
        t[-2] = int(0.9 * Mi)  # one near-full group
    so = torch.zeros(B + 1, dtype=torch.int32, device=device)
    so[1:] = torch.cumsum(t, 0).to(torch.int32)
    return so


def _make_inputs(
    B, D, Kout, Mi, regime="uniform", seed=SEED, device="cuda", sparsity=0.95
):
    """jagged (L, K), dense (B, K, N), dOut (L, N), seq_offsets (B+1,)."""
    torch.manual_seed(0)
    N, K = Kout, D
    seq_offsets = _make_seq_offsets(B, Mi, regime, seed, device, sparsity=sparsity)
    L = int(seq_offsets[-1].item())
    jagged = torch.randn(max(L, 1), K, dtype=dtypes.bf16, device=device)
    dense = torch.randn(B, K, N, dtype=dtypes.bf16, device=device)
    d_out = torch.randn(max(L, 1), N, dtype=dtypes.bf16, device=device)
    return jagged, dense, d_out, seq_offsets, L, N, K


def run_torch(jagged, dense, d_out, seq_offsets, N, K):
    """Eager dJagged (L, K), dDense (B, K, N), dBias (B, N). Reference only."""
    L = jagged.shape[0]
    B = dense.shape[0]
    d_jagged = torch.zeros((L, K), dtype=dtypes.bf16, device=jagged.device)
    d_dense = torch.zeros((B, K, N), dtype=dtypes.bf16, device=jagged.device)
    d_bias = torch.zeros((B, N), dtype=dtypes.bf16, device=jagged.device)
    for b in range(B):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            go = d_out[s:e].float()
            d_jagged[s:e] = (go @ dense[b].float().t()).to(dtypes.bf16)
            d_dense[b] = (jagged[s:e].float().t() @ go).to(dtypes.bf16)
            d_bias[b] = go.sum(0).to(dtypes.bf16)
    return d_jagged, d_dense, d_bias


def _flops_bytes(component, L, B, D, N):
    f_jag = 2.0 * L * D * N
    f_db = 2.0 * L * D * N
    m_jag = (L * N + B * D * N + L * D) * 2
    m_db = (L * D + L * N + B * D * N + B * N) * 2
    if component == "jagged":
        return f_jag, m_jag
    if component == "dense_bias":
        return f_db, m_db
    return f_jag + f_db, m_jag + m_db


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _component_cos(component, got, ref):
    d_jagged_ref, d_dense_ref, d_bias_ref = ref
    if component == "jagged":
        return _cos(got, d_jagged_ref)
    if component == "dense_bias":
        dd, db = got
        return min(_cos(dd, d_dense_ref), _cos(db, d_bias_ref))
    dj, dd, db = got
    return min(_cos(dj, d_jagged_ref), _cos(dd, d_dense_ref), _cos(db, d_bias_ref))


def _build_triton_fn(jagged, dense, d_out, seq_offsets, B, Mi, N, K, component):
    """Triton closures for perf timing.

    Hoists ``d_jagged`` and ``d_dense`` (the buffers the upstream API accepts as
    outputs) so timed iterations do not pay per-iter ``empty_like`` allocation.
    ``triton_jagged_dense_bmm_add_bwd_dense_bias`` (``elementwise=False``) still
    allocates ``d_bias`` inside the function on every call; that cost is upstream
    API behavior, not a benchmark artifact. FlyDSL hoists ``d_bias`` because its
    launcher accepts a caller-provided buffer.
    """
    so64 = seq_offsets.to(torch.int64)
    d_jagged_out = torch.empty_like(jagged)
    d_dense_out = torch.empty_like(dense)

    def run_jagged():
        return triton_jagged_dense_bmm_add_bwd_jagged(
            Mi, so64, d_jagged_out, dense, d_out, K, B, N
        )

    def run_dense_bias():
        return triton_jagged_dense_bmm_add_bwd_dense_bias(
            Mi, so64, jagged, d_dense_out, B, K, N, d_out, False
        )

    if component == "jagged":
        return run_jagged
    if component == "dense_bias":
        return run_dense_bias

    def run_all():
        return run_jagged(), *run_dense_bias()

    return run_all


def _build_flydsl_fn(jagged, dense, d_out, seq_offsets, B, Mi, N, K, component):
    """FlyDSL closures for perf timing.

    ``component == "all"`` uses the production dispatched wrapper (alloc + all
    launches). ``jagged`` / ``dense_bias`` use raw launchers over pre-allocated
    buffers for kernel-only timing. ``jagged`` is directly comparable to Triton;
    ``dense_bias`` is not fully comparable because Triton allocates ``d_bias``
    internally each call (see ``_build_triton_fn``).
    """
    _bwd.configure_dim(K)
    block_m = _bwd.BLOCK_M
    split = _bwd.SPLIT
    device = jagged.device
    total_rows = jagged.shape[0]
    stream = torch.cuda.current_stream()

    if component == "all":

        def run_all():
            return jagged_dense_bmm_bwd_dispatched(
                jagged,
                dense,
                d_out,
                seq_offsets,
                n_groups=B,
                max_seq_len=Mi,
                stream=stream,
            )

        return run_all

    t_d_out = flyc.from_dlpack(d_out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    if component == "jagged":
        dense_kn = dense.reshape(B * K, N).contiguous()
        d_jagged = torch.zeros(
            total_rows + block_m, K, dtype=dtypes.bf16, device=device
        )
        t_dj = flyc.from_dlpack(d_jagged).mark_layout_dynamic(
            leading_dim=1, divisibility=8
        )

        def run_jagged():
            _bwd.grad_jagged(t_dj, t_d_out, dense_kn, seq_offsets, B, Mi, stream=stream)
            return d_jagged[:total_rows]

        return run_jagged

    d_dense = torch.zeros(B, K, N, dtype=dtypes.bf16, device=device)
    d_dense_v = d_dense.view(B * K, N)
    dense_partials = torch.zeros(B * split * K, N, dtype=torch.float32, device=device)
    d_bias = torch.zeros(B, N, dtype=dtypes.bf16, device=device)
    bias_partials = torch.zeros(B * split, N, dtype=torch.float32, device=device)
    t_jagged = flyc.from_dlpack(jagged).mark_layout_dynamic(
        leading_dim=1, divisibility=8
    )

    def run_dense_bias():
        _bwd.grad_dense_bias(
            d_dense_v,
            d_bias,
            t_jagged,
            t_d_out,
            seq_offsets,
            dense_partials,
            bias_partials,
            B,
            Mi,
            stream=stream,
        )
        return d_dense, d_bias

    return run_dense_bias


@benchmark()
def jdbba_bwd(B, D, Kout, Mi, regime, component, seed=SEED, sparsity=0.95):
    jagged, dense, d_out, seq_offsets, L, N, K = _make_inputs(
        B, D, Kout, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    ref = run_torch(jagged, dense, d_out, seq_offsets, N, K)

    candidates = {
        "flydsl": _build_flydsl_fn(
            jagged, dense, d_out, seq_offsets, B, Mi, N, K, component
        ),
    }
    if _HAS_TRITON:
        candidates["triton"] = _build_triton_fn(
            jagged, dense, d_out, seq_offsets, B, Mi, N, K, component
        )

    flops, nbytes = _flops_bytes(component, L, B, D, N)
    ret = {"gfx": get_gfx(), "L": L}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        min_cos = _component_cos(component, got, ref)
        assert (
            min_cos > _COS_THRESH
        ), f"{name}: jdbba_bwd ({component}, {regime}) cos={min_cos:.5f}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(0.0, 1.0 - min_cos)
    return ret


def _eager_grads(jagged, dense, bias, seq_offsets, grad_out, B):
    j = jagged.float().detach().clone().requires_grad_(True)
    d = dense.float().detach().clone().requires_grad_(True)
    bi = bias.float().detach().clone().requires_grad_(True)
    go = grad_out.float()
    loss = j.new_zeros(())
    for b in range(B):
        s, e = int(seq_offsets[b]), int(seq_offsets[b + 1])
        if e > s:
            out_b = j[s:e] @ d[b] + bi[b][None, :]
            loss = loss + (out_b * go[s:e]).sum()
    loss.backward()
    return j.grad, d.grad, bi.grad


def _run_autograd_case(D, B, Mi, regime, *, seed=SEED, sparsity=0.95, label=""):
    from aiter.ops.flydsl import jagged_dense_bmm_autograd

    jagged, dense, grad_out, seq_offsets, L, N, _K = _make_inputs(
        B, D, D, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    bias = torch.randn(B, N, dtype=dtypes.bf16, device=jagged.device)

    jf = jagged.detach().clone().requires_grad_(True)
    df = dense.detach().clone().requires_grad_(True)
    bf = bias.detach().clone().requires_grad_(True)
    out = jagged_dense_bmm_autograd(
        jf,
        df,
        bf,
        seq_offsets,
        n_groups=B,
        max_seq_len=Mi,
        uniform_seqlen=(regime == "uniform"),
    )
    out.backward(grad_out)
    torch.cuda.synchronize()

    gj_e, gd_e, gb_e = _eager_grads(jagged, dense, bias, seq_offsets, grad_out, B)
    c_j, c_d, c_b = _cos(jf.grad, gj_e), _cos(df.grad, gd_e), _cos(bf.grad, gb_e)
    ok = min(c_j, c_d, c_b) > _COS_THRESH
    tag = f"{label}[autograd] B={B} D={D} Mi={Mi} {regime:6s} L={L}"
    return (
        ok,
        f"[{'PASS' if ok else 'FAIL'}] {tag}  grad cos(dJ={c_j:.5f}, dD={c_d:.5f}, dB={c_b:.5f})",
    )


def _run_case(D, B, Mi, regime, *, seed=SEED, sparsity=0.95, label=""):
    from aiter.ops.flydsl.jagged_dense_bmm_bwd_dispatch import resolve_config

    jagged, dense, d_out, seq_offsets, L, N, K = _make_inputs(
        B, D, D, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    dj, dd, db = jagged_dense_bmm_bwd_dispatched(
        jagged, dense, d_out, seq_offsets, n_groups=B, max_seq_len=Mi
    )
    torch.cuda.synchronize()
    rj, rd, rb = run_torch(jagged, dense, d_out, seq_offsets, N, K)
    c_dj, c_dd, c_db = _cos(dj, rj), _cos(dd, rd), _cos(db, rb)
    ok = min(c_dj, c_dd, c_db) > _COS_THRESH
    cfg = resolve_config(n_groups=B, reduction_k=D, output_n=D, max_seq_len=Mi)
    bw = _bwd.build_backward(
        D,
        split=cfg["split"],
        gj_stages_a=cfg["gj_stages_a"],
        coarsen_m=cfg["coarsen_m"],
    )
    tag = (
        f"{label}B={B} D={D} Mi={Mi} {regime:6s} L={L} "
        f"gj={cfg['gj_stages_a']} split={bw.split}"
    )
    return (
        ok,
        f"[{'PASS' if ok else 'FAIL'}] {tag}  cos(dJ={c_dj:.5f}, dD={c_dd:.5f}, dB={c_db:.5f})",
    )


def _run_reduce_path_case(D, B, Mi, regime, *, seed=SEED, sparsity=0.95):
    jagged, dense, d_out, seq_offsets, L, N, K = _make_inputs(
        B, D, D, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    dj, dd, db = jagged_dense_bmm_bwd_dispatched(
        jagged, dense, d_out, seq_offsets, n_groups=B, max_seq_len=Mi, split=2
    )
    torch.cuda.synchronize()
    rj, rd, rb = run_torch(jagged, dense, d_out, seq_offsets, N, K)
    c_dj, c_dd, c_db = _cos(dj, rj), _cos(dd, rd), _cos(db, rb)
    ok = min(c_dj, c_dd, c_db) > _COS_THRESH
    ncols = (N + min(N, 256) - 1) // min(N, 256)
    tag = f"[reduce-path] B={B} D={D} Mi={Mi} {regime:6s} split=2 NRED_COL_TILES={ncols} L={L}"
    return (
        ok,
        f"[{'PASS' if ok else 'FAIL'}] {tag}  cos(dJ={c_dj:.5f}, dD={c_dd:.5f}, dB={c_db:.5f})",
    )


def _worker(D: int) -> int:
    from aiter.ops.flydsl.jagged_dense_bmm_bwd_dispatch import resolve_config

    ok = True

    expected_gj = 1 if D <= 256 else 2
    cfg = resolve_config(n_groups=1024, reduction_k=D, output_n=D, max_seq_len=7680)
    res_ok = cfg["gj_stages_a"] == expected_gj
    ok &= res_ok
    print(
        f"[{'PASS' if res_ok else 'FAIL'}] resolve winner D={D}: gj_stages_a={cfg['gj_stages_a']} "
        f"(expected {expected_gj})"
    )

    expected_split = 2 if D <= 256 else 1
    bw = _bwd.build_backward(D, split=None, gj_stages_a=expected_gj, coarsen_m=None)
    split_ok = bw.split == expected_split
    ok &= split_ok
    print(
        f"[{'PASS' if split_ok else 'FAIL'}] build_backward(D={D}).split={bw.split} "
        f"(expected {expected_split})"
    )

    for B, Mi, regime in _HEADLINE:
        case_ok, msg = _run_case(D, B, Mi, regime)
        ok &= case_ok
        print(msg)

    if D == 512:
        for B, Mi, regime in _LARGE_B_OVERFLOW:
            case_ok, msg = _run_case(D, B, Mi, regime, label="[large-B] ")
            ok &= case_ok
            print(msg)

        r = _REGRESSION
        case_ok, msg = _run_case(
            D,
            r["B"],
            r["Mi"],
            r["regime"],
            seed=r["seed"],
            sparsity=r["sparsity"],
            label="[regression] ",
        )
        ok &= case_ok
        print(msg)

    for B, Mi, regime in _HEADLINE:
        case_ok, msg = _run_autograd_case(D, B, Mi, regime)
        ok &= case_ok
        print(msg)

    if D == 512:
        r = _REGRESSION
        case_ok, msg = _run_autograd_case(
            D,
            r["B"],
            r["Mi"],
            r["regime"],
            seed=r["seed"],
            sparsity=r["sparsity"],
            label="[regression] ",
        )
        ok &= case_ok
        print(msg)

    print(f"\nD={D} RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def _backend_ready() -> bool:
    if not _HAS_FLYDSL or not torch.cuda.is_available():
        return False
    return get_gfx() in SUPPORTED_GFX


_requires_backend = pytest.mark.skipif(
    not _backend_ready(),
    reason="jdbba backward requires flydsl on gfx942/gfx950",
)


@_requires_backend
@pytest.mark.parametrize("D", [256, 512])
def test_jdbba_bwd_dispatch_worker(D):
    assert _worker(D) == 0


@_requires_backend
@pytest.mark.parametrize("D", [512, 384])
def test_jdbba_bwd_reduce_path(D):
    ok, msg = _run_reduce_path_case(D, 120, 512, "genrec")
    assert ok, msg


@_requires_backend
def test_jdbba_bwd_multi_device():
    """Backward on cuda:1 while current device is cuda:0 (stream/device dispatch).

    Runs in a subprocess so a HIP context-pollution failure cannot leak into the
    rest of the session. Validates ``current_stream(device=jagged.device)`` in
    ``jagged_dense_bmm_bwd_dispatched`` when tensor device != current device.

    If the FlyDSL runtime pins to device 0 internally, the subprocess may raise
    ``hipErrorInvalidDevice`` and the test is marked xfail (mirrors FMHA).
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    import subprocess
    import textwrap

    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_REPO_ROOT!r})

        import torch
        import torch.nn.functional as F
        from aiter import dtypes
        from aiter.ops.flydsl import jagged_dense_bmm_bwd_dispatched

        torch.cuda.set_device(0)
        dev1 = torch.device("cuda", 1)
        B, D, Mi = 32, 256, 128
        N, K = D, D
        seq_offsets = torch.arange(
            0, (B + 1) * Mi, Mi, dtype=torch.int32, device=dev1
        )
        L = int(seq_offsets[-1].item())
        jagged = torch.randn(L, K, dtype=dtypes.bf16, device=dev1)
        dense = torch.randn(B, K, N, dtype=dtypes.bf16, device=dev1)
        d_out = torch.randn(L, N, dtype=dtypes.bf16, device=dev1)

        dj, dd, db = jagged_dense_bmm_bwd_dispatched(
            jagged, dense, d_out, seq_offsets, n_groups=B, max_seq_len=Mi
        )
        torch.cuda.synchronize(dev1)
        for t in (dj, dd, db):
            assert t.device == dev1, f"expected {{dev1}} got {{t.device}}"

        d_jagged = torch.zeros((L, K), dtype=dtypes.bf16, device=dev1)
        d_dense = torch.zeros((B, K, N), dtype=dtypes.bf16, device=dev1)
        d_bias = torch.zeros((B, N), dtype=dtypes.bf16, device=dev1)
        for b in range(B):
            s = int(seq_offsets[b].item())
            e = int(seq_offsets[b + 1].item())
            if e > s:
                go = d_out[s:e].float()
                d_jagged[s:e] = (go @ dense[b].float().t()).to(dtypes.bf16)
                d_dense[b] = (jagged[s:e].float().t() @ go).to(dtypes.bf16)
                d_bias[b] = go.sum(0).to(dtypes.bf16)

        cos_thresh = {_COS_THRESH}
        for name, got, ref in (
            ("dJagged", dj, d_jagged),
            ("dDense", dd, d_dense),
            ("dBias", db, d_bias),
        ):
            c = F.cosine_similarity(
                got.float().flatten(), ref.float().flatten(), dim=0
            ).item()
            assert c > cos_thresh, f"{{name}} cos={{c:.6f}}"

        print("MULTI_DEVICE_OK", flush=True)
        """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "MULTI_DEVICE_OK" in proc.stdout:
        return
    if "hipErrorInvalidDevice" in combined or "invalid device ordinal" in combined:
        pytest.xfail(
            "FlyDSL runtime pins to device 0; wrapper-level stream/device "
            "selection is in place but underlying runtime does not honor it"
        )
    raise AssertionError(
        f"multi-device subprocess failed unexpectedly:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def main(argv=None) -> int:
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "jagged_dense_bmm_bwd unsupported on %s; skipping", get_gfx()
        )
        return 0
    if not _HAS_FLYDSL:
        aiter.logger.warning(
            "flydsl unavailable; skipping jagged_dense_bmm_bwd perf sweep"
        )
        return 0
    if not _HAS_TRITON:
        aiter.logger.warning(
            "generative-recommenders Triton baseline unavailable (%s); running flydsl-only",
            _TRITON_ERR,
        )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="jagged_dense_bmm backward test + perf sweep",
    )
    parser.add_argument(
        "--worker-d",
        type=int,
        default=None,
        help="run correctness cases for a single D (debug; skips perf sweep)",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=_PERF_SHAPES,
        help="perf shapes as B,D,Kout e.g. -s 120,256,256",
    )
    parser.add_argument(
        "-r",
        "--regime",
        type=str,
        choices=["uniform", "skew", "genrec"],
        nargs="*",
        default=["uniform"],
        help="sequence-length distribution for perf sweep",
    )
    parser.add_argument(
        "-c",
        "--component",
        type=str,
        choices=["jagged", "dense_bias", "all"],
        nargs="*",
        default=["all"],
        help="backward component(s) to score in perf sweep",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="RNG seed for skew/genrec perf sweeps",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.95,
        help="genrec regime: scale M_i ~ Uniform(1,Mi) by this factor (clamped >=1)",
    )
    parser.add_argument(
        "-mi",
        type=int,
        default=_PERF_MI,
        help="max_seq_len for perf sweep (default deployment Mi)",
    )
    args = parser.parse_args(argv)

    if args.worker_d is not None:
        return _worker(args.worker_d)

    rows = []
    for component, regime, (B, D, Kout) in itertools.product(
        args.component, args.regime, args.shapes
    ):
        rows.append(
            jdbba_bwd(
                B,
                D,
                Kout,
                args.mi,
                regime,
                component,
                seed=args.seed,
                sparsity=args.sparsity,
            )
        )
    _summarize("jagged_dense_bmm_bwd", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

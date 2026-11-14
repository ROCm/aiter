# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the standalone FlyDSL GDN K6 output kernel.

Both K6 implementations are fed the SAME operands, so any difference is the
output stage's own. The Triton ``chunk_fwd_o_opt_vk`` is the gold standard.

``h`` and ``v_new`` are drawn at random rather than produced by a K5 reference:
K6 consumes them as opaque inputs, so a real K5 run would only narrow the value
range this exercises. What does have to be faithful is ``g``, which K6 both
exponentiates per row and differences per (row, column) pair -- it is built the
way ``chunk_local_cumsum`` builds it, restarting the running sum at every chunk
boundary and at every sequence boundary under varlen.
"""

from __future__ import annotations

import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL K6 tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.gdn_o_kernels import chunk_fwd_o_flydsl, flydsl_k6_supported
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Unable to import FlyDSL K6 dependencies: {exc}",
        allow_module_level=True,
    )

# Both kernels consume identical bf16 operands and accumulate in f32, so they
# should agree far more tightly than either agrees with an fp32 reference. The
# residual is MFMA-vs-tl.dot accumulation order plus one bf16 rounding of A.
_K6_RMSE_TOL = 2e-2

_BT = 64


def _rmse_ratio(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a, b = actual.float(), ref.float()
    return (
        torch.sqrt(((a - b) ** 2).mean()) / (torch.sqrt((b**2).mean()) + 1e-12)
    ).item()


def _chunk_local_cumsum(decay: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    """Running gate sum, restarted at each chunk and each sequence boundary.

    ``decay`` is [H, T_flat] of per-token (negative) log gates; the result is
    what K6 reads as ``g``.
    """
    g = torch.empty_like(decay)
    start = 0
    for seq_len in seq_lens:
        for c0 in range(0, seq_len, _BT):
            lo = start + c0
            hi = start + min(c0 + _BT, seq_len)
            g[:, lo:hi] = decay[:, lo:hi].cumsum(dim=1)
        start += seq_len
    return g


def _make_k6_inputs(
    *,
    H,
    Hg,
    K,
    V,
    seq_lens,
    varlen,
    decay_per_token=0.15,
    seed=0,
    device="cuda",
):
    """The operand set K6 takes, without running K5 to get it."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    if varlen:
        B, T = 1, sum(seq_lens)
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(0)),
            dtype=torch.long,
            device=device,
        )
        # Chunks are packed across sequences, each sequence padded up to BT.
        NT = sum((s + _BT - 1) // _BT for s in seq_lens)
        cumsum_lens = seq_lens
    else:
        assert len(set(seq_lens)) == 1, "a dense batch needs one common length"
        B, T = len(seq_lens), seq_lens[0]
        cu_seqlens = None
        NT = (T + _BT - 1) // _BT
        cumsum_lens = [T]

    T_flat = T
    q = torch.randn(B, T, Hg, K, dtype=dtype, device=device)
    k = torch.randn(B, T, Hg, K, dtype=dtype, device=device)
    # v_new is head-major, the layout K5 drains it in.
    v = torch.randn(B, H, T_flat, V, dtype=dtype, device=device)
    # h is the per-chunk state snapshot, [V, K] per (chunk, head).
    h_lead = (NT,) if varlen else (B, NT)
    h = torch.randn(*h_lead, H, V, K, dtype=dtype, device=device)

    decay = -torch.rand(B * H, T_flat, dtype=torch.float32, device=device) * (
        2.0 * decay_per_token
    )
    g = _chunk_local_cumsum(decay, cumsum_lens * B if not varlen else cumsum_lens)
    g = g.view(B, H, T_flat).contiguous()

    o = torch.empty(B, T, H, V, dtype=dtype, device=device)
    return dict(q=q, k=k, v=v, h=h, g=g, o=o, cu_seqlens=cu_seqlens, K=K, V=V)


def _run_both(inp, *, scale=None, use_exp2=False, BV=None):
    common = dict(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        h=inp["h"],
        g=inp["g"],
        scale=scale if scale is not None else inp["K"] ** -0.5,
        cu_seqlens=inp["cu_seqlens"],
        use_exp2=use_exp2,
    )
    o_triton = torch.empty_like(inp["o"])
    chunk_fwd_o_opt_vk(o=o_triton, **common)

    o_flydsl = torch.empty_like(inp["o"])
    chunk_fwd_o_flydsl(o=o_flydsl, BV=BV, **common)
    return o_flydsl, o_triton


def _skip_if_unsupported(inp, BV):
    if not flydsl_k6_supported(
        q=inp["q"], h=inp["h"], K=inp["K"], V=inp["V"], BV=BV, chunk_size=_BT
    ):
        pytest.skip("the FlyDSL K6 kernel does not support this device / shape")


@pytest.mark.parametrize(
    "H,Hg",
    [(12, 12), (4, 2), (24, 24)],  # MHA, GQA, MHA-wide
)
@pytest.mark.parametrize(
    "seq_lens,varlen",
    [
        ([512], False),  # dense, single batch, BT-aligned
        ([500], False),  # dense, tail chunk
        ([512, 512, 512], False),  # dense, multi-batch (exercises the b*NT stride)
        ([512, 512], True),  # varlen, aligned
        ([640, 384, 500], True),  # varlen, tail chunk on the last sequence
    ],
)
@pytest.mark.parametrize("BV", [32, 64, 128])
def test_k6_matches_triton(H, Hg, seq_lens, varlen, BV):
    """FlyDSL K6 matches Triton K6 on identical operands."""
    inp = _make_k6_inputs(H=H, Hg=Hg, K=128, V=128, seq_lens=seq_lens, varlen=varlen)
    _skip_if_unsupported(inp, BV)

    o_flydsl, o_triton = _run_both(inp, BV=BV)

    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


@pytest.mark.parametrize("K,V", [(64, 64), (128, 128), (64, 128), (128, 64)])
def test_k6_head_dims(K, V):
    """Non-square and narrow head dimensions."""
    inp = _make_k6_inputs(H=8, Hg=8, K=K, V=V, seq_lens=[512, 384], varlen=True)
    BV = min(64, V)
    _skip_if_unsupported(inp, BV)

    o_flydsl, o_triton = _run_both(inp, BV=BV)

    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


@pytest.mark.parametrize("use_exp2", [False, True])
def test_k6_exp2(use_exp2):
    """The log2-scaled gate path agrees with Triton's."""
    inp = _make_k6_inputs(H=12, Hg=12, K=128, V=128, seq_lens=[512, 384], varlen=True)
    _skip_if_unsupported(inp, 64)
    if use_exp2:
        # USE_EXP2 reads g as already log2-scaled; scale the gate both kernels
        # see, so the comparison is of exp2-vs-exp evaluation and not of two
        # different gates.
        inp["g"] = (inp["g"] * math.log2(math.e)).contiguous()

    o_flydsl, o_triton = _run_both(inp, use_exp2=use_exp2)

    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


def test_k6_strong_decay():
    """A steep gate must not blow up the pair term.

    The pair gate exp(g_i - g_j) is only ever evaluated under the causal mask,
    where g_i <= g_j bounds it by 1. This is the regression guard for that: a
    telescoping formulation keyed on g_last would evaluate exp(+320) here and
    overflow f32.
    """
    inp = _make_k6_inputs(
        H=12, Hg=12, K=128, V=128, seq_lens=[512], varlen=False, decay_per_token=5.0
    )
    _skip_if_unsupported(inp, 64)
    assert inp["g"].min() < -100, "the strong-decay case did not build a steep gate"

    o_flydsl, o_triton = _run_both(inp)

    assert torch.isfinite(o_flydsl).all(), "FlyDSL K6 produced non-finite output"
    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


def test_k6_no_gate():
    """g=None degenerates to the ungated projection."""
    inp = _make_k6_inputs(H=8, Hg=8, K=128, V=128, seq_lens=[512, 384], varlen=True)
    _skip_if_unsupported(inp, 64)
    inp["g"] = None

    o_flydsl, o_triton = _run_both(inp)

    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


if __name__ == "__main__":
    rows = []
    for H, Hg in ((12, 12), (4, 2)):
        for seq_lens, varlen in (
            ([512], False),
            ([500], False),
            ([640, 384, 500], True),
        ):
            for BV in (32, 64, 128):
                inp = _make_k6_inputs(
                    H=H, Hg=Hg, K=128, V=128, seq_lens=seq_lens, varlen=varlen
                )
                try:
                    o_f, o_t = _run_both(inp, BV=BV)
                    rmse = _rmse_ratio(o_f, o_t)
                    status = "PASS" if rmse < _K6_RMSE_TOL else "FAIL"
                except Exception as exc:  # noqa: BLE001
                    rmse, status = float("nan"), f"ERROR: {type(exc).__name__}"
                rows.append((f"{H}/{Hg}", str(seq_lens), BV, rmse, status))

    print(f"\n{'H/Hg':>6} {'seq_lens':>20} {'BV':>4} {'RMSE':>10} status")
    for hhg, sl, bv, rmse, status in rows:
        print(f"{hhg:>6} {sl:>20} {bv:>4} {rmse:10.3e} {status}")
    n_pass = sum(1 for r in rows if r[4] == "PASS")
    print(f"\n{n_pass}/{len(rows)} passed")

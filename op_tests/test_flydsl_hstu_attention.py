# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for FlyDSL HSTU attention forward.

Covers both forward kernels and the dispatch between them: the ping-pong
kernel (gfx942, bf16, head_dim == hidden_dim in (64, 128)) and the fallback.
"""

import argparse
import csv
import itertools
import math

import pandas as pd
import pytest
import torch
from flydsl.runtime.device import get_rocm_arch

import aiter
import aiter.ops.flydsl.hstu_attention as hstu_kernels
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl.hstu_attention import (
    _validate_inputs,
    flydsl_hstu_attention_fwd,
    fwd_pp_tiles,
)
from aiter.ops.flydsl.kernels.hstu.hstu_attention_fwd_pp import (
    _bias_slot_check_range,
    _bias_slot_map,
    _check_bias_slots,
    build_hstu_attention_fwd_pp,
    validate_hstu_attention_fwd_pp,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

_DEFAULT_DEVICE = torch.device("cuda")


def _load_torch_hstu_reference():
    """Import the triton-side torch reference under either import root."""

    try:
        from triton_tests.utils.hstu_attention_ref import torch_hstu_attention
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not (
            missing == "triton_tests"
            or missing.startswith(("triton_tests.", "op_tests."))
            or missing == "op_tests"
        ):
            raise

        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from op_tests.triton_tests.utils.hstu_attention_ref import torch_hstu_attention

    return torch_hstu_attention


def _generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(1)

    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len = int((2 * sparsity - 1.0) * max_seq_len)
    else:
        min_seq_len = 0
        max_seq_len = int(2 * sparsity * max_seq_len)

    return torch.randint(
        low=min_seq_len,
        high=max_seq_len,
        size=(size,),
        device=device,
        dtype=torch.int,
    )


def _apply_sl(lengths: torch.Tensor, alpha: float, max_seq_len: int) -> torch.Tensor:
    threshold = int(max_seq_len ** (alpha / 2.0))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    return torch.where(users_to_sample, threshold, lengths)


def generate_hstu_attn_inputs(
    batch_size: int,
    max_seq_len: int,
    sparsity: float,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    target_size: int,
    dtype: torch.dtype,
    device: torch.device = _DEFAULT_DEVICE,
    seed: int = 1001,
    sl_alpha: float = 2.0,
):
    torch.manual_seed(seed)

    lengths = _generate_sparse_seq_len(batch_size, max_seq_len, sparsity, device)
    if sl_alpha > 0:
        lengths = _apply_sl(lengths, sl_alpha, max_seq_len=max_seq_len)

    num_targets = None
    if target_size > 0:
        num_targets = torch.randint(
            1,
            target_size + 1,
            (batch_size,),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        num_targets = torch.where(num_targets > lengths, lengths, num_targets)

    seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total_len = int(seq_offsets[-1].item())

    x = torch.empty(
        (total_len, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=device,
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

    return q.contiguous(), k.contiguous(), v.contiguous(), seq_offsets, num_targets


# Validated on gfx942 and gfx950
HSTU_SUPPORTED_GFX = ["gfx942", "gfx950"]


def _pp_supported(
    num_heads, head_dim, hidden_dim, causal, max_attn_len, contextual_seq_len,
    has_targets, alpha, dtype,
) -> bool:  # fmt: skip
    """Whether the ping-pong forward serves this problem at its dispatch tile."""
    if dtype != torch.bfloat16:
        return False
    try:
        validate_hstu_attention_fwd_pp(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            causal=causal,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            has_targets=has_targets,
            alpha=alpha,
            dtype_str="bf16",
            arch=get_gfx_runtime(),
            **fwd_pp_tiles(head_dim, max_attn_len),
        )
    except ValueError:
        return False
    return True


def _hstu_fwd_flops(
    seq_offsets: torch.Tensor, heads: int, attn_dim: int, hidden_dim: int
) -> float:
    """Lower-triangular mask: no x2 factor on matmul FLOPs."""
    total_flops = 0.0
    seq_num = seq_offsets.shape[0] - 1
    for i in range(seq_num):
        length = int(seq_offsets[i + 1].item() - seq_offsets[i].item())
        total_flops += length * length * (attn_dim + hidden_dim) * heads
    return total_flops


def _hstu_fwd_bytes(
    seq_offsets: torch.Tensor,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    elem_size: int,
) -> int:
    seq_num = seq_offsets.shape[0] - 1
    total_bytes = 0
    for i in range(seq_num):
        length = int(seq_offsets[i + 1].item() - seq_offsets[i].item())
        total_bytes += length * (attn_dim + length + hidden_dim) * heads * elem_size
    return total_bytes


# Padded torch ref builds [B, H, N, N]; skip it above this size in perf sweeps.
_REF_MAX_QK_BYTES = 8 * 1024**3


def _padded_qk_bytes(batch_size, max_seq_len, num_heads, elem_size) -> int:
    return batch_size * num_heads * max_seq_len * max_seq_len * elem_size


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_hstu_attention.py"
)
@benchmark()
def test_flydsl_hstu_attention_perf(
    batch_size,
    max_seq_len,
    sparsity,
    num_heads,
    head_dim,
    hidden_dim,
    target_size,
    dtype,
    max_attn_len=0,
    contextual_seq_len=0,
    causal=True,
    seed=1001,
):
    torch_hstu_attention = _load_torch_hstu_reference()

    torch.cuda.empty_cache()
    device = torch.device("cuda")
    alpha = 1.0 / head_dim * 10000

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        heads=num_heads,
        attn_dim=head_dim,
        hidden_dim=hidden_dim,
        target_size=target_size,
        dtype=dtype,
        device=device,
        seed=seed,
    )

    def fwd(pingpong):
        return lambda: flydsl_hstu_attention_fwd(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            max_attn_len,
            contextual_seq_len,
            pingpong=pingpong,
        )

    # `default` is what the dispatch picks; the other two pin one kernel each.
    candidates = {"default": fwd(None), "fallback": fwd(False)}
    # nan cells where the ping-pong's gate rejects the shape.
    if _pp_supported(
        num_heads, head_dim, hidden_dim, causal, max_attn_len, contextual_seq_len,
        num_targets is not None, alpha, dtype,
    ):  # fmt: skip
        candidates["pingpong"] = fwd(True)

    # Production shapes OOM the dense padded ref; correctness is covered by
    # the parametrized pytest cases at smaller max_seq_len.
    out_ref = None
    qk_bytes = _padded_qk_bytes(batch_size, max_seq_len, num_heads, q.element_size())
    if qk_bytes <= _REF_MAX_QK_BYTES:
        out_ref = (
            torch_hstu_attention(
                max_seq_len,
                alpha,
                q,
                k,
                v,
                seq_offsets,
                causal,
                dropout_pr=0.0,
                training=False,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                min_full_attn_seq_len=0,
            )
            * max_seq_len
        )

    flops = _hstu_fwd_flops(seq_offsets, num_heads, head_dim, hidden_dim)
    nbytes = _hstu_fwd_bytes(
        seq_offsets, num_heads, head_dim, hidden_dim, q.element_size()
    )
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = float("nan")
        if out_ref is not None:
            err = checkAllclose(
                out_ref,
                out * max_seq_len,
                rtol=0,
                atol=1e-3,
                msg=f"{name}: flydsl_hstu_attention_fwd",
            )
            assert err == 0, f"{name}: {err:.2%} of elements outside atol"
        ret[f"{name} us"] = round(us, 3)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(nbytes / us / 1e6, 3)
        ret[f"{name} err"] = err
    return ret


@pytest.mark.parametrize(
    "batch_size,max_seq_len,sparsity,"
    "max_attn_len,contextual_seq_len,target_size,"
    "attn_dim,hidden_dim,causal",
    [
        # Tiny smoke test
        (1, 64, 1.0, 0, 0, 0, 128, 128, True),
        # causal
        (4, 512, 0.5, 0, 0, 0, 128, 128, True),
        # target_size > 0
        (4, 512, 0.5, 0, 0, 20, 128, 128, True),
        # max_attn_len > 0
        (4, 512, 0.5, 64, 0, 0, 128, 128, True),
        # contextual_seq_len > 0 (causal only)
        (4, 512, 0.5, 0, 64, 0, 128, 128, True),
        # symmetric and dims %64 != 0
        (4, 512, 0.5, 0, 0, 0, 96, 96, True),
        # not symmetric
        (4, 512, 0.5, 0, 0, 0, 128, 64, True),
        # not symmetric and dims %64 != 0
        (4, 512, 0.5, 0, 0, 0, 96, 192, True),
        # batch*heads (3*4=12) not divisible by NUM_GRID_GROUPS
        (3, 512, 0.5, 0, 0, 0, 128, 128, True),
        # non-causal
        # full bidirectional attention
        (4, 512, 0.5, 0, 0, 0, 128, 128, False),
        # non-causal + targets (abs(dist) over target-clamped ids — interaction untested elsewhere)
        (4, 512, 0.5, 0, 0, 20, 128, 128, False),
        # non-causal symmetric window (|q - col| <= max_attn_len)
        (4, 512, 0.5, 64, 0, 0, 128, 128, False),
        # non-causal + contextual + symmetric window (contextual OR reopens row 0 to prefix)
        (4, 512, 0.5, 64, 64, 0, 128, 128, False),
    ],
)
def test_flydsl_hstu_attention(
    batch_size: int,
    max_seq_len: int,
    sparsity: float,
    max_attn_len: int,
    contextual_seq_len: int,
    target_size: int,
    attn_dim: int,
    hidden_dim: int,
    causal: bool,
    heads: int = 4,
    dtype=torch.bfloat16,
):
    # The torch reference lives on the triton side; import it lazily so that
    # merely importing this module stays triton-free for pytest collection.
    torch_hstu_attention = _load_torch_hstu_reference()

    torch.cuda.empty_cache()

    alpha = 1.0 / attn_dim * 10000

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        heads=heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        target_size=target_size,
        dtype=dtype,
        device=torch.device("cuda"),
    )

    def flydsl_attn():
        return flydsl_hstu_attention_fwd(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            max_attn_len,
            contextual_seq_len,
        )

    def torch_attn():
        return torch_hstu_attention(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=0,
        )

    out = flydsl_attn() * max_seq_len
    out_ref = torch_attn() * max_seq_len
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=0)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA device"
)


def _qkv(batch=2, tokens=8, heads=4, attn_dim=128, hidden_dim=128, device="cuda"):
    q = torch.zeros((tokens, heads, attn_dim), dtype=torch.bfloat16, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros((tokens, heads, hidden_dim), dtype=torch.bfloat16, device=device)
    seq_offsets = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    return q, k, v, seq_offsets


@requires_cuda
def test_validate_inputs_ok():
    q, k, v, seq_offsets = _qkv(batch=2, heads=4, attn_dim=128, hidden_dim=96)
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_inputs(
        q, k, v, seq_offsets, None, max_seq_len=8
    )

    actual = (batch, num_heads, head_dim, hidden_dim, dtype_str)
    expected = (2, 4, 128, 96, "bf16")
    assert actual == expected


def test_validate_inputs_rejects_cpu_tensors():
    q, k, v, seq_offsets = _qkv(device="cpu")
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, None, max_seq_len=8)


@requires_cuda
def test_validate_inputs_rejects_qk_shape_mismatch():
    q, k, v, seq_offsets = _qkv()
    k = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2] + 8), dtype=q.dtype, device=q.device
    )
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, None, max_seq_len=8)


@requires_cuda
def test_validate_inputs_rejects_wrong_rank():
    q, k, v, seq_offsets = _qkv()
    with pytest.raises(ValueError):
        _validate_inputs(q.reshape(-1), k, v, seq_offsets, None, max_seq_len=8)


@requires_cuda
def test_validate_inputs_rejects_dtype_mismatch():
    q, k, v, seq_offsets = _qkv()
    v = v.to(torch.float16)
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, None, max_seq_len=8)


@requires_cuda
def test_validate_inputs_rejects_num_targets_length_mismatch():
    q, k, v, seq_offsets = _qkv(batch=2)
    num_targets = torch.zeros(3, dtype=torch.int64, device=q.device)  # != batch
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, num_targets, max_seq_len=8)


# --------------------------------------------------------------------------- #
# Ping-pong forward and the default dispatch (gfx942 only)
# --------------------------------------------------------------------------- #

requires_pp = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx942",
    reason="the ping-pong forward is gfx942-only",
)

# (causal, max_attn_len, contextual_seq_len, target_size). Windows start the
# KV sweep past tile 0 (W=1 is the narrowest band); contextual + window checks
# the window bound in effective IDs (the whole prefix is ID 0).
_PP_MASKS = [
    (True, 0, 0, 20),  # hstu: causal + targets
    (True, 0, 0, 0),  # causal
    (True, 512, 0, 20),  # windowed, the dispatcher's (256,32) case
    (True, 100, 0, 20),
    (True, 1, 0, 20),
    (True, 0, 64, 0),  # contextual
    (True, 20, 50, 0),  # contextual + window
    (True, 100, 700, 20),
    (False, 0, 0, 0),  # non-causal
]


def _no_tuned_rows(monkeypatch):
    """Dispatch as if the tuned CSV were empty, so only the heuristic decides."""
    monkeypatch.setattr(hstu_kernels, "_tuned_config_map", lambda tuned_file=None: {})


def _spy_pp_builds(monkeypatch) -> list:
    """Record every ping-pong build so a test can tell it was really dispatched.

    `_compile_launcher` is lru_cached across tests, so it is cleared first:
    otherwise a shape compiled by an earlier test returns without building.
    """
    builds = []
    real = hstu_kernels.build_hstu_attention_fwd_pp

    def spy(**kwargs):
        builds.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hstu_kernels, "build_hstu_attention_fwd_pp", spy)
    hstu_kernels._compile_launcher.cache_clear()
    return builds


def _built_tiles(builds) -> list:
    return [(b["block_m"], b["block_n"]) for b in builds]


def _pp_case(max_seq_len, max_attn_len, contextual_seq_len, target_size, causal,
             attn_dim=64, hidden_dim=64, batch_size=5, heads=4,
             dtype=torch.bfloat16):  # fmt: skip
    torch_hstu_attention = _load_torch_hstu_reference()
    alpha = 1.0 / attn_dim * 10000
    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=0.75,
        heads=heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        target_size=target_size,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    args = (max_seq_len, alpha, q, k, v, seq_offsets, causal, num_targets,
            max_attn_len, contextual_seq_len)  # fmt: skip
    ref = torch_hstu_attention(
        *args[:7],
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=0,
    )
    return args, ref


def _assert_matches_ref(out, ref, max_seq_len):
    torch.testing.assert_close(out * max_seq_len, ref * max_seq_len, atol=1e-3, rtol=0)


@requires_pp
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_m,block_n", [(512, 32), (256, 32)])
@pytest.mark.parametrize(
    "causal,max_attn_len,contextual_seq_len,target_size", _PP_MASKS
)
def test_flydsl_hstu_attention_pingpong(
    monkeypatch, head_dim, block_m, block_n, causal, max_attn_len,
    contextual_seq_len, target_size,
):  # fmt: skip
    builds = _spy_pp_builds(monkeypatch)
    # 1100 > 2*512: every tile height sees a sequence spanning several Q tiles
    # and a partial last tile.
    args, ref = _pp_case(
        1100, max_attn_len, contextual_seq_len, target_size, causal,
        attn_dim=head_dim, hidden_dim=head_dim,
    )  # fmt: skip
    out = flydsl_hstu_attention_fwd(
        *args, block_m=block_m, block_n=block_n, pingpong=True
    )
    assert _built_tiles(builds) == [(block_m, block_n)]
    _assert_matches_ref(out, ref, args[0])


# (head_dim, max_seq_len, causal, max_attn_len, contextual_seq_len, expected
# ping-pong tile or None for the fallback), per `_fwd_pp_preferred`.
_DEFAULT_DISPATCH = [
    (64, 4096, True, 0, 0, (512, 32)),
    (64, 2048, True, 0, 0, None),  # too short for the 512-row tile
    (64, 4096, True, 512, 0, (256, 32)),
    (64, 4096, True, 256, 0, None),  # window too narrow
    (128, 1024, True, 0, 0, (256, 32)),  # head_dim 128 wins at any length
    (128, 1024, True, 64, 0, (256, 32)),
    (64, 4096, True, 0, 64, None),  # contextual prefix
    (64, 4096, False, 0, 0, None),  # non-causal
]


@requires_pp
@pytest.mark.parametrize(
    "head_dim,max_seq_len,causal,max_attn_len,contextual_seq_len,expected",
    _DEFAULT_DISPATCH,
)
def test_flydsl_hstu_attention_default_dispatch(
    monkeypatch, head_dim, max_seq_len, causal, max_attn_len, contextual_seq_len,
    expected,
):  # fmt: skip
    _no_tuned_rows(monkeypatch)
    builds = _spy_pp_builds(monkeypatch)
    args, ref = _pp_case(
        max_seq_len, max_attn_len, contextual_seq_len, 20, causal,
        attn_dim=head_dim, hidden_dim=head_dim, batch_size=3, heads=2,
    )  # fmt: skip
    out = flydsl_hstu_attention_fwd(*args)
    assert _built_tiles(builds) == ([expected] if expected else [])
    _assert_matches_ref(out, ref, args[0])


def _tuned_rows_for(args, rows):
    """A tuned map for exactly this problem, keyed as `_compile_launcher` looks it up.

    `rows` maps kernel -> (block_m, block_n, num_waves, waves_per_eu), fastest
    kernel first.
    """
    max_seq_len, _, q, _, v, seq_offsets, _, num_targets, max_attn_len, ctx = args
    key = hstu_kernels._problem_key(
        hstu_kernels._GPU_ARCH, "bf16", q.shape[1], q.shape[2], v.shape[2],
        seq_offsets.numel() - 1, max_seq_len, max_attn_len > 0, ctx > 0,
        num_targets is not None,
    )  # fmt: skip
    names = ("block_m", "block_n", "num_waves", "waves_per_eu")
    return {key: {kernel: dict(zip(names, t)) for kernel, t in rows.items()}}


def _spy_fallback_builds(monkeypatch) -> list:
    builds = []
    real = hstu_kernels.build_hstu_attention_fwd

    def spy(**kwargs):
        builds.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hstu_kernels, "build_hstu_attention_fwd", spy)
    return builds


@requires_pp
def test_flydsl_hstu_attention_tuned_rows_select_kernel(monkeypatch):
    """Tuned rows override the heuristic both ways, each kernel with its own tiles."""
    pp_builds = _spy_pp_builds(monkeypatch)
    fallback_builds = _spy_fallback_builds(monkeypatch)
    # Without rows the heuristic picks the ping-pong at (512,32) here.
    args, ref = _pp_case(4096, 0, 0, 20, True, batch_size=3, heads=2)

    def run_with(rows, **kwargs):
        tuned = _tuned_rows_for(args, rows)
        monkeypatch.setattr(hstu_kernels, "_tuned_config_map", lambda f=None: tuned)
        hstu_kernels._compile_launcher.cache_clear()
        pp_builds.clear()
        fallback_builds.clear()
        _assert_matches_ref(flydsl_hstu_attention_fwd(*args, **kwargs), ref, args[0])
        return _built_tiles(pp_builds), _built_tiles(fallback_builds)

    both_fallback_first = {"fallback": (64, 32, 4, 2), "pingpong": (384, 32, 8, 2)}
    both_pp_first = {"pingpong": (384, 32, 8, 2), "fallback": (64, 32, 4, 2)}
    assert run_with(both_fallback_first) == ([], [(64, 32)])
    assert run_with(both_pp_first) == ([(384, 32)], [])
    # An override still gets the other kernel's tuned tiles, not the defaults.
    assert run_with(both_pp_first, pingpong=False) == ([], [(64, 32)])
    assert run_with(both_fallback_first, pingpong=True) == ([(384, 32)], [])


@requires_pp
def test_flydsl_hstu_attention_pingpong_overrides(monkeypatch):
    """pingpong=False and explicit tiles both keep a ping-pong shape on the fallback."""
    _no_tuned_rows(monkeypatch)
    builds = _spy_pp_builds(monkeypatch)
    args, ref = _pp_case(4096, 0, 0, 20, True, batch_size=3, heads=2)
    _assert_matches_ref(flydsl_hstu_attention_fwd(*args, pingpong=False), ref, args[0])
    # Explicit tiles are the fallback kernel's unless pingpong=True.
    out = flydsl_hstu_attention_fwd(*args, block_m=128, block_n=32)
    _assert_matches_ref(out, ref, args[0])
    assert builds == []


@requires_pp
@pytest.mark.parametrize(
    "attn_dim,hidden_dim,dtype,match",
    [
        (128, 64, torch.bfloat16, "head_dim == hidden_dim"),
        (256, 256, torch.bfloat16, "head_dim in"),
        (64, 64, torch.float16, "bf16"),
    ],
)
def test_flydsl_hstu_attention_pingpong_unsupported(
    monkeypatch, attn_dim, hidden_dim, dtype, match
):
    """An explicit request raises; the default silently runs the fallback."""
    _no_tuned_rows(monkeypatch)
    builds = _spy_pp_builds(monkeypatch)
    args, ref = _pp_case(
        4096, 0, 0, 20, True, attn_dim=attn_dim, hidden_dim=hidden_dim,
        batch_size=3, heads=2, dtype=dtype,
    )  # fmt: skip
    with pytest.raises(ValueError, match=match):
        flydsl_hstu_attention_fwd(*args, pingpong=True)
    _assert_matches_ref(flydsl_hstu_attention_fwd(*args), ref, args[0])
    assert builds == []


def _ref_for(args):
    max_seq_len, alpha, q, k, v, seq_offsets, causal, num_targets, w, ctx = args
    return _load_torch_hstu_reference()(
        max_seq_len, alpha, q, k, v, seq_offsets, causal, dropout_pr=0.0,
        training=False, num_targets=num_targets, max_attn_len=w,
        contextual_seq_len=ctx, min_full_attn_seq_len=0,
    )  # fmt: skip


@requires_cuda
@pytest.mark.parametrize("pingpong", [None, False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_flydsl_hstu_attention_noncontiguous_v(pingpong, head_dim):
    """A non-contiguous v, after and before a contiguous one of the same shape.

    The output used to inherit v's strides: the ping-pong stores whole rows and
    wrote the wrong elements, and either kernel's cached compiled handle was
    wrong for the second layout.
    """
    if pingpong and get_gfx_runtime() != "gfx942":
        pytest.skip("the ping-pong forward is gfx942-only")
    # N=4096 so the default dispatch picks the ping-pong at both head dims.
    args, ref = _pp_case(
        4096, 0, 0, 20, True, attn_dim=head_dim, hidden_dim=head_dim,
        batch_size=3, heads=2,
    )  # fmt: skip
    v = args[4]
    v_nc = v.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    assert torch.equal(v_nc, v) and not v_nc.is_contiguous()
    for vv in (v, v_nc, v):
        out = flydsl_hstu_attention_fwd(*args[:4], vv, *args[5:], pingpong=pingpong)
        assert out.is_contiguous()
        _assert_matches_ref(out, ref, args[0])


@requires_pp
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("max_attn_len", [0, 512])
@pytest.mark.parametrize("k_val", [-1.0, -16.0])
def test_flydsl_hstu_attention_pingpong_large_negative_scores(
    head_dim, max_attn_len, k_val
):
    """silu's negative tail must vanish: alpha*s far below the sigmoid table's
    range gates to 0 (the table used to clamp to sigmoid(16) without a window,
    so p grew linearly with |s|)."""
    n, h = 2048, 2
    seq_offsets = torch.tensor([0, n, n + 1500], dtype=torch.int64, device="cuda")
    shape = (int(seq_offsets[-1]), h, head_dim)
    q = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.full(shape, k_val, dtype=torch.bfloat16, device="cuda")
    v = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    args = (n, 1.0, q, k, v, seq_offsets, True, None, max_attn_len, 0)
    out = flydsl_hstu_attention_fwd(*args, pingpong=True)
    _assert_matches_ref(out, _ref_for(args), n)


@requires_pp
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_m", [512, 256])
def test_flydsl_hstu_attention_pingpong_exact_cancellation(head_dim, block_m):
    """q.k == 0 exactly, so every output is 0. The score must stay fp32 until
    the gate: rounding a prescaled Q to bf16 broke the cancellation (0.23)."""
    n = 1024
    seq_offsets = torch.tensor([0, n], dtype=torch.int64, device="cuda")
    q = torch.zeros((n, 1, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    q[:, 0, 0], q[:, 0, 1] = 1.0, 1.625
    k[:, 0, 0], k[:, 0, 1] = 208.0, -128.0
    v = torch.ones_like(q)
    args = (n, 1.0, q, k, v, seq_offsets, True, None, 0, 0)
    out = flydsl_hstu_attention_fwd(*args, block_m=block_m, block_n=32, pingpong=True)
    _assert_matches_ref(out, _ref_for(args), n)


@requires_pp
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("max_attn_len", [0, 512])
def test_flydsl_hstu_attention_pingpong_unit_scale_inputs(head_dim, max_attn_len):
    """N(0,1) q/k/v with alpha = 1/sqrt(d), so scores are O(1), unlike the
    +-0.01 inputs elsewhere. The ping-pong's mean error must match the
    fallback's; with Q prescaled in bf16 it was ~1.6x."""
    n, h, b = 4096, 2, 4
    g = torch.Generator(device="cuda").manual_seed(0)
    lengths = torch.randint(1, n + 1, (b,), device="cuda", generator=g)
    lengths = (lengths.float() * 0.95).long().clamp(min=1)
    seq_offsets = torch.zeros(b + 1, dtype=torch.int64, device="cuda")
    seq_offsets[1:] = torch.cumsum(lengths, 0)
    shape = (int(seq_offsets[-1]), h, head_dim)
    q, k, v = (
        torch.randn(shape, device="cuda", generator=g).to(torch.bfloat16)
        for _ in range(3)
    )
    args = (n, head_dim**-0.5, q, k, v, seq_offsets, True, None, max_attn_len, 0)
    ref = _ref_for(args).float()
    err = {
        pp: (flydsl_hstu_attention_fwd(*args, pingpong=pp).float() - ref).abs()
        for pp in (True, False)
    }
    assert float(err[True].mean()) <= 1.2 * float(err[False].mean()), (
        float(err[True].mean()),
        float(err[False].mean()),
    )
    # 1 bf16 ULP of the output's scale on top of the fallback's max error.
    ulp = 2.0 ** (math.floor(math.log2(float(ref.abs().max()))) - 7)
    assert float(err[True].max()) <= float(err[False].max()) + ulp


@pytest.mark.parametrize("kv_subtiles", [2])
def test_bias_slot_map_matches_patterns(kv_subtiles):
    """The window-bias slot map, checked over the FULL v range for many
    windows. The builder checks only the edge ranges, which this shows is
    enough; checking everything there was O(max_attn_len) per build."""
    for w in [*range(1, 1025), 2048, 4096, 16385]:
        _, _, jt, *_ = _bias_slot_map(w, kv_subtiles)
        _check_bias_slots(w, kv_subtiles, range(-4, jt + kv_subtiles + 8))
    for w in (10**6, 10**9):
        assert len(_bias_slot_check_range(w, kv_subtiles)) < 64


@requires_pp
def test_pingpong_build_huge_window():
    """Building with a huge window must not walk the whole window (this used to
    take ~1.5 h at max_attn_len=1e9). Build only; nothing is launched."""
    build_hstu_attention_fwd_pp(
        num_heads=2, head_dim=64, hidden_dim=64, causal=True,
        max_attn_len=10**9, contextual_seq_len=0, has_targets=False,
        alpha=1.0, dtype_str="bf16", block_m=256, block_n=32, num_waves=8,
        waves_per_eu=2,
    )  # fmt: skip


# --------------------------------------------------------------------------- #
# Tuned CSV loading
# --------------------------------------------------------------------------- #


def _row(**overrides) -> dict:
    row = {
        "arch": get_rocm_arch(),
        "dtype": "bf16",
        "num_heads": 4,
        "head_dim": 128,
        "hidden_dim": 128,
        "batch": 256,
        "max_seq_len": 1024,
        "has_window": "False",
        "has_contextual": "False",
        "has_targets": "False",
        "kernel": "fallback",
        "duration": 1.0,
        "block_m": 128,
        "block_n": 64,
        "num_waves": 4,
        "waves_per_eu": 2,
    }
    row.update(overrides)
    return row


def _write_csv(path, rows) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hstu_kernels._CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def test_tuned_csv_is_picked_up(tmp_path):
    path = _write_csv(tmp_path / "tuned.csv", [_row()])

    config_map = hstu_kernels._tuned_config_map(path)

    assert len(config_map) == 1
    (rows,) = config_map.values()
    assert rows == {
        "fallback": {"block_m": 128, "block_n": 64, "num_waves": 4, "waves_per_eu": 2}
    }


def test_tuned_csv_rejects_unknown_kernel(tmp_path):
    path = _write_csv(
        tmp_path / "tuned.csv",
        [_row(kernel="mystery"), _row(kernel="pingpong", max_seq_len=4096)],
    )

    config_map = hstu_kernels._tuned_config_map(path)

    (rows,) = config_map.values()
    assert list(rows) == ["pingpong"]


def test_tuned_csv_keeps_both_kernels_fastest_first(tmp_path):
    """Rows for both kernels on one shape: both are kept, the faster first."""
    path = _write_csv(
        tmp_path / "tuned.csv",
        [
            _row(kernel="fallback", duration=3.0),
            _row(kernel="pingpong", duration=2.0, block_m=512, num_waves=8),
        ],
    )

    (rows,) = hstu_kernels._tuned_config_map(path).values()
    assert list(rows) == ["pingpong", "fallback"]
    assert rows["pingpong"]["block_m"] == 512
    assert rows["fallback"]["block_m"] == 128


def test_tuned_csv_missing_file_returns_empty(tmp_path):
    assert hstu_kernels._tuned_config_map(str(tmp_path / "does_not_exist.csv")) == {}


def test_tuned_csv_best_duration_wins(tmp_path):
    path = _write_csv(
        tmp_path / "tuned.csv",
        [
            _row(duration=5.0, block_m=64),
            _row(duration=1.0, block_m=256),
        ],
    )

    config_map = hstu_kernels._tuned_config_map(path)

    (rows,) = config_map.values()
    assert rows["fallback"]["block_m"] == 256


def main():
    if get_gfx() not in HSTU_SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_hstu_attention unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL HSTU attention perf sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp16],
        help="dtype sweep (bf16, fp16)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[120],
        help="batch_size sweep",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        nargs="*",
        default=[16384],
        help="max_seq_len sweep",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="*",
        default=[0.475],
        help="sparsity sweep",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="*",
        default=[4],
        help="num_heads sweep",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        nargs="*",
        default=[64],
        help="head_dim (attn_dim) sweep",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="*",
        default=[0],
        help="hidden_dim sweep (0 -> head_dim)",
    )
    parser.add_argument(
        "--max-attn-len",
        type=int,
        nargs="*",
        default=[0],
        help="max_attn_len sweep",
    )
    parser.add_argument(
        "--contextual-seq-len",
        type=int,
        nargs="*",
        default=[0],
        help="contextual_seq_len sweep",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs="*",
        default=[300],
        help="target_size sweep",
    )
    args = parser.parse_args()

    rows = []
    for (
        dtype,
        batch_size,
        max_seq_len,
        sparsity,
        num_heads,
        head_dim,
        hidden_dim,
        max_attn_len,
        contextual_seq_len,
        target_size,
    ) in itertools.product(
        args.dtype,
        args.batch,
        args.max_seq_len,
        args.sparsity,
        args.num_heads,
        args.head_dim,
        args.hidden_dim,
        args.max_attn_len,
        args.contextual_seq_len,
        args.target_size,
    ):
        rows.append(
            test_flydsl_hstu_attention_perf(
                batch_size,
                max_seq_len,
                sparsity,
                num_heads,
                head_dim,
                hidden_dim or head_dim,
                target_size,
                dtype,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
            )
        )
    aiter.logger.info(
        "flydsl_hstu_attention summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()

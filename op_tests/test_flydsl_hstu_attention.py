# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for FlyDSL HSTU attention forward."""

import argparse
import csv
import itertools

import pandas as pd
import pytest
import torch

import aiter
import aiter.ops.flydsl.hstu_attention_kernels as hstu_kernels
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hstu_attention_kernels import (
    _validate_inputs,
    flydsl_hstu_attention_fwd,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

_DEFAULT_DEVICE = torch.device("cuda")


def _load_torch_hstu_reference():
    """Import the triton-side torch reference under either import root."""

    try:
        from triton_tests.utils.hstu_attention_ref import torch_hstu_attention
    except ModuleNotFoundError as exc:
        if exc.name != "triton_tests":
            raise
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

    out, us = run_perftest(flydsl_attn)
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
        err = checkAllclose(
            out_ref,
            out * max_seq_len,
            rtol=0,
            atol=1e-3,
            msg="flydsl_hstu_attention_fwd",
        )
    else:
        # Production shapes OOM the dense padded ref; correctness is covered by
        # the parametrized pytest cases at smaller max_seq_len.
        err = float("nan")

    flops = _hstu_fwd_flops(seq_offsets, num_heads, head_dim, hidden_dim)
    nbytes = _hstu_fwd_bytes(
        seq_offsets, num_heads, head_dim, hidden_dim, q.element_size()
    )
    return {
        "gfx": get_gfx(),
        "flydsl us": round(us, 3),
        "flydsl TFLOPS": round(flops / us / 1e6, 1),
        "flydsl TB/s": round(nbytes / us / 1e6, 3),
        "flydsl err": err,
    }


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
# Tuned CSV loading
# --------------------------------------------------------------------------- #


def _row(**overrides) -> dict:
    row = {
        "arch": hstu_kernels._GPU_ARCH,
        "dtype": "bf16",
        "num_heads": 4,
        "head_dim": 128,
        "hidden_dim": 128,
        "batch": 256,
        "max_seq_len": 1024,
        "has_window": "False",
        "has_contextual": "False",
        "has_targets": "False",
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
    (config,) = config_map.values()
    assert config == {"block_m": 128, "block_n": 64, "num_waves": 4, "waves_per_eu": 2}


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

    (config,) = config_map.values()
    assert config["block_m"] == 256


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

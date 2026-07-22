# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import itertools

import aiter
import pandas as pd
import pytest
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

import aiter.ops.flydsl.hstu_attention_kernels as hstu_kernels
from aiter.ops.flydsl.hstu_attention_kernels import (
    flydsl_hstu_attention_fwd,
    _validate_inputs,
)

# Every card this op is built/validated for; the kernel's validate() enforces the
# same set, but gate the whole perf sweep here so it is a clean no-op elsewhere.
SUPPORTED_GFX = ["gfx942", "gfx950"]


# --------------------------------------------------------------------------- #
# Input generation (shared with op_benchmarks/flydsl/bench_hstu_attn.py)
# --------------------------------------------------------------------------- #
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


def generate_hstu_attn_inputs(
    batch_size: int,
    max_seq_len: int,
    sparsity: float,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    target_size: int,
    dtype: torch.dtype,
    device: torch.device = torch.device("cuda"),
    seed: int = 1001,
):
    """Build (q, k, v, seq_offsets, num_targets) for an HSTU attention problem.

    Self-contained (no triton-side dependencies) so both the tests here and the
    FlyDSL benchmark can share a single input generator.
    """
    torch.manual_seed(seed)  # for reproducibility

    # sparsity-controlled jagged lengths (mean ~= sparsity * max_seq_len). The
    # old power-law resampling step (apply_SL) is intentionally dropped: ported
    # with alpha=0.2 it collapsed every sequence to ~2 tokens, which made the
    # perf roofline meaningless. Realistic lengths here match the mvonstra
    # recsys_harness convention used to validate this kernel.
    lengths = _generate_sparse_seq_len(batch_size, max_seq_len, sparsity, device)

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


# --------------------------------------------------------------------------- #
# Reference (oracle only — never timed, never in the summary table)
# --------------------------------------------------------------------------- #
def run_torch(
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
):
    # The torch reference lives on the triton side; import it lazily so merely
    # importing this module (e.g. from the benchmark) stays triton-free.
    from op_tests.triton_tests.utils.hstu_attention_ref import torch_hstu_attention

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


def _roofline(seq_offsets, heads, attn_dim, hidden_dim, elem_size):
    """FLOPs and HBM bytes the op actually does, for TFLOPS / TB-s.

    Causal HSTU is two GEMMs per (query, key) pair: Q·K^T contracts attn_dim and
    P·V contracts hidden_dim. Valid pairs ~ length^2/2 (lower-triangular) and each
    pair is a multiply-add (2 flops), so the causal 1/2 and the MAC 2 cancel:
        FLOPs ~ sum_b length^2 * (attn_dim + hidden_dim) * heads.
    A fused kernel never materialises the L*L score matrix in HBM, so the traffic
    is just q + k (attn_dim) read and v read + out written (hidden_dim):
        bytes ~ sum_b length * heads * 2*(attn_dim + hidden_dim) * elem_size.
    """
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.float64).cpu()
    flops = float((lengths * lengths).sum()) * (attn_dim + hidden_dim) * heads
    total_tokens = float(lengths.sum())
    nbytes = total_tokens * heads * 2 * (attn_dim + hidden_dim) * elem_size
    return flops, nbytes


# --------------------------------------------------------------------------- #
# Perf + correctness sweep (aiter-op-test format)
# --------------------------------------------------------------------------- #
@benchmark()
def test_hstu_attention_fwd(
    batch_size,
    max_seq_len,
    sparsity,
    num_heads,
    head_dim,
    hidden_dim,
    max_attn_len,
    contextual_seq_len,
    target_size,
    dtype,
):
    causal = True
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
    )

    # torch reference is the oracle only — compute once, compare against it, never time it.
    ref = run_torch(
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

    candidates = {
        "flydsl": lambda: flydsl_hstu_attention_fwd(
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
        ),
    }

    flops, nbytes = _roofline(seq_offsets, num_heads, head_dim, hidden_dim, q.element_size())

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        # Output carries the 1/N normalisation; rescale both sides to O(1) so the
        # bf16/f16 tolerance is meaningful, then compare in fp32.
        err = checkAllclose(
            (ref.to(dtypes.fp32) * max_seq_len),
            (out.to(dtypes.fp32) * max_seq_len),
            atol=1e-3,
            rtol=0,
            msg=f"{name}: hstu_attention_fwd",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# @benchmark wraps the fn so its call args become table columns; it is driven by
# main() / the correctness test below, not collected as a bare pytest test.
test_hstu_attention_fwd.__test__ = False


# --------------------------------------------------------------------------- #
# Correctness (drives the @benchmark fn and asserts the kernel matches torch)
# --------------------------------------------------------------------------- #
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA device"
)


@requires_cuda
@pytest.mark.parametrize(
    "batch_size,max_seq_len,sparsity,"
    "max_attn_len,contextual_seq_len,target_size,"
    "head_dim,hidden_dim",
    [
        (256, 1024, 0.5, 0, 0, 0, 128, 128),
        # target_size > 0
        (256, 1024, 0.5, 0, 0, 20, 128, 128),
        # max_attn_len > 0
        (256, 1024, 0.5, 64, 0, 0, 128, 128),
        # contextual_seq_len > 0
        (256, 1024, 0.5, 0, 64, 0, 128, 128),
        # symmetric and dims %64 != 0
        (256, 1024, 0.5, 0, 0, 0, 96, 96),
        # not symmetric
        (256, 1024, 0.5, 0, 0, 0, 128, 64),
        # not symmetric and dims %64 != 0
        (256, 1024, 0.5, 0, 0, 0, 96, 192),
    ],
)
def test_flydsl_hstu_attention(
    batch_size,
    max_seq_len,
    sparsity,
    max_attn_len,
    contextual_seq_len,
    target_size,
    head_dim,
    hidden_dim,
    num_heads=4,
    dtype=torch.bfloat16,
):
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"hstu_attention_fwd unsupported on {get_gfx()}")

    torch.cuda.empty_cache()

    ret = test_hstu_attention_fwd(
        batch_size,
        max_seq_len,
        sparsity,
        num_heads,
        head_dim,
        hidden_dim,
        max_attn_len,
        contextual_seq_len,
        target_size,
        dtype,
    )
    assert ret["flydsl err"] == 0, ret


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
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
        q, k, v, seq_offsets, None
    )

    actual = (batch, num_heads, head_dim, hidden_dim, dtype_str)
    expected = (2, 4, 128, 96, "bf16")
    assert actual == expected


def test_validate_inputs_rejects_cpu_tensors():
    q, k, v, seq_offsets = _qkv(device="cpu")
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, None)


@requires_cuda
def test_validate_inputs_rejects_qk_shape_mismatch():
    q, k, v, seq_offsets = _qkv()
    k = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2] + 8), dtype=q.dtype, device=q.device
    )
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, None)


@requires_cuda
def test_validate_inputs_rejects_wrong_rank():
    q, k, v, seq_offsets = _qkv()
    with pytest.raises(ValueError):
        _validate_inputs(q.reshape(-1), k, v, seq_offsets, None)


@requires_cuda
def test_validate_inputs_rejects_dtype_mismatch():
    q, k, v, seq_offsets = _qkv()
    v = v.to(torch.float16)
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, None)


@requires_cuda
def test_validate_inputs_rejects_num_targets_length_mismatch():
    q, k, v, seq_offsets = _qkv(batch=2)
    num_targets = torch.zeros(3, dtype=torch.int64, device=q.device)  # != batch
    with pytest.raises(ValueError):
        _validate_inputs(q, k, v, seq_offsets, num_targets)


# --------------------------------------------------------------------------- #
# Tuned CSV loading
# --------------------------------------------------------------------------- #
def _row(**overrides) -> dict:
    row = dict(
        arch=hstu_kernels._GPU_ARCH,
        dtype="bf16",
        num_heads=4,
        head_dim=128,
        hidden_dim=128,
        batch=256,
        max_seq_len=1024,
        has_window="False",
        has_contextual="False",
        has_targets="False",
        duration=1.0,
        block_m=128,
        block_n=64,
        num_waves=4,
        waves_per_eu=2,
    )
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
    assert config == dict(block_m=128, block_n=64, num_waves=4, waves_per_eu=2)


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


# --------------------------------------------------------------------------- #
# Perf sweep entry point (markdown summary table)
# --------------------------------------------------------------------------- #
def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "hstu_attention_fwd unsupported on %s; skipping perf sweep", get_gfx()
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
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="Data type, e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[128],
        help="Batch size (num sequences). e.g.: -b 128",
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096],
        help="max_seq_len per sequence. e.g.: --seq 1024 2048",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="*",
        default=[0.5],
        help="Jagged sequence-length sparsity in [0,1]. e.g.: --sparsity 0.5",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=[4],
        help="num_heads. e.g.: --heads 4",
    )
    parser.add_argument(
        "-s",
        "--dims",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(128, 128), (64, 64)],
        help="(head_dim, hidden_dim) pairs. e.g.: -s 128,128 64,64",
    )
    parser.add_argument(
        "--max_attn_len",
        type=int,
        nargs="*",
        default=[0],
        help="Sliding-window length (0 = full causal). e.g.: --max_attn_len 0 256",
    )
    parser.add_argument(
        "--contextual_seq_len",
        type=int,
        nargs="*",
        default=[0],
        help="Contextual prefix length (0 = off). e.g.: --contextual_seq_len 0 64",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs="*",
        default=[0],
        help="Max target-tail size (0 = no targets). e.g.: --target_size 0 20",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for (
            batch,
            seq,
            sparsity,
            heads,
            (head_dim, hidden_dim),
            max_attn_len,
            contextual_seq_len,
            target_size,
        ) in itertools.product(
            args.batch,
            args.seq,
            args.sparsity,
            args.heads,
            args.dims,
            args.max_attn_len,
            args.contextual_seq_len,
            args.target_size,
        ):
            df.append(
                test_hstu_attention_fwd(
                    batch,
                    seq,
                    sparsity,
                    heads,
                    head_dim,
                    hidden_dim,
                    max_attn_len,
                    contextual_seq_len,
                    target_size,
                    dtype,
                )
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "flydsl_hstu_attention_fwd summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()

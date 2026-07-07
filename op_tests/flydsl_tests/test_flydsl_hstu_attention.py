import pytest
import torch
import csv

import aiter.ops.flydsl.hstu_attention_kernels as hstu_kernels
from aiter.ops.flydsl.hstu_attention_kernels import (
    flydsl_hstu_attention_fwd,
    _validate_inputs,
)


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
    device: torch.device = torch.device("cuda"),
    seed: int = 1001,
):
    """Build (q, k, v, seq_offsets, num_targets) for an HSTU attention problem.

    Self-contained (no triton-side dependencies) so both the tests here and the
    FlyDSL benchmark can share a single input generator.
    """
    torch.manual_seed(seed)  # for reproducibility

    lengths = _generate_sparse_seq_len(batch_size, max_seq_len, sparsity, device)
    lengths = _apply_sl(lengths, 0.2, max_seq_len=max_seq_len)

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


@pytest.mark.parametrize(
    "batch_size,max_seq_len,sparsity,"
    "max_attn_len,contextual_seq_len,target_size,"
    "attn_dim,hidden_dim",
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
    batch_size: int,
    max_seq_len: int,
    sparsity: float,
    max_attn_len: int,
    contextual_seq_len: int,
    target_size: int,
    attn_dim: int,
    hidden_dim: int,
    heads: int = 4,
    dtype = torch.bfloat16,
):
    # The torch reference lives on the triton side; import it lazily so that
    # merely importing this module (e.g. from the benchmark) stays triton-free.
    from op_tests.triton_tests.utils.hstu_attention_ref import torch_hstu_attention

    torch.cuda.empty_cache()  

    causal = True
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


import pytest
import os
import torch
from typing import Optional, Tuple
import torch.nn.functional as F

from aiter.ops.triton.hstu_attention import (
    _AttentionFunction,
)


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# generate inputs
def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(1)  # for reproducibility

    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        max_seq_len: int = max_seq_len
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)

    return torch.randint(
        low=min_seq_len,
        high=max_seq_len,
        size=(size,),
        device=device,
        dtype=torch.int,
    )


def apply_SL(
    lengths: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    threshold = int(max_seq_len ** (alpha / 2.0))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    lengths = torch.where(users_to_sample, threshold, lengths)
    return lengths

def sanity_check_attention(
    max_seq_len: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    dropout_pr: float,
    seq2_offsets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
) -> None:
    Z = seq_offsets.numel() - 1
    _, H, _ = q.shape
    torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
    torch._assert(q.dim() == 3, "q must be 3-D")
    torch._assert(k.shape == q.shape, "k must be the same shape as q")
    torch._assert(v.dim() == 3, "v must be 3-D")
    torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
    torch._assert(v.shape[1] == H, "wrong v shape[1]")
    if attn_bias is not None:
        assert seq2_offsets is not None
        torch._assert(attn_bias.dim() == 1, "attn_bias must be 1-D")
        torch._assert(
            seq2_offsets is not None,
            "must have seq2_offsets when using attn_bias",
        )
        torch._assert(seq2_offsets.dim() == 1, "seq2_offsets must be 1-D")
    if max_attn_len is not None:
        torch._assert(max_attn_len > 0, "max_attn_len must be larger than 0")
    if invalid_attn_mask_type != "lower_triangular":
        torch._assert(
            contextual_seq_len == 0,
            "user context mask not supported on non-lower triangular mask",
        )
    torch._assert(q.is_cuda, "q must be CUDA tensor")
    torch._assert(k.is_cuda, "k must be CUDA tensor")
    torch._assert(v.is_cuda, "v must be CUDA tensor")
    torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
    if attn_bias is not None:
        torch._assert(attn_bias.is_cuda, "attn_bias must be CUDA tensor")
        assert seq2_offsets is not None
        torch._assert(seq2_offsets.is_cuda, "seq2_offsets must be CUDA tensor")
    torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")


# functions related to reference torch implementation
# create attention mask
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return valid_attn_mask


# convert sequence input from jagged format to padded dense format 
def jagged_to_padded_dense(q: torch.Tensor, offsets: torch.Tensor, max_seq_len: int, padding_value):
    assert len(q.shape) == 2, "q needs to be 2-dim tensor"
    L, D = q.shape
    B = offsets.shape[0] - 1
    padded_shape = (B, max_seq_len, D)
    padded_q = torch.full(padded_shape, padding_value, dtype=q.dtype, device=q.device)
    for i in range(B):
        s = offsets[i]
        e = offsets[i + 1]
        padded_q[i][0: e - s] = q[s : e]

    return padded_q


# pad sequence according to max sequence len
def pad_sequence(q: torch.Tensor, seq_offsets: torch.Tensor, N: int, padding_value):
    L, D = q.shape
    padded_q = jagged_to_padded_dense(
        q.reshape(L, D),
        offsets = seq_offsets,
        max_seq_len = N,
        padding_value = 0.0
    )

    return padded_q


def qkv_to_padded_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = pad_sequence(q.reshape(L, H * D), seq_offsets, N, 0.0).view(-1, N, H, D).transpose(1, 2)
    padded_k = pad_sequence(k.reshape(L, H * D), seq_offsets, N, 0.0).view(-1, N, H, D).transpose(1, 2)
    padded_v = pad_sequence(v.reshape(L, H * D), seq_offsets, N, 0.0).view(-1, N, H, D).transpose(1, 2)
    
    return padded_q, padded_k, padded_v


# convert sequences from dense format to jagged format
def dense_to_jagged(seq: torch.Tensor, offsets: torch.Tensor, L: int):
    B, N, HV = seq.shape
    assert L == offsets[-1], f"jagged dim mismatch {offsets[-1]} != {L}!"
    out = torch.empty((L, HV), dtype=seq.dtype, device=seq.device)

    for i in range(B):
        s = offsets[i]
        e = offsets[i + 1]
        out[s : e] = seq[i][0 : e - s]

    return out


#torch hstu reference implementation
def torch_hstu_attention(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = qkv_to_padded_dense(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )
    # raise NotImplementedError(valid_attn_mask[0, :, :].to(torch.int32))
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        seq_offsets,
        L,
    ).view(L, H, V)


@pytest.mark.parametrize("batch_size, max_seq_len, sparsity",
                         [(512, 3072, 0.366),
                          (512, 512, 0.97)])
def test_hstu_attention(
    batch_size: int,
    max_seq_len: int,  # for repro
    sparsity: float,  # for repro
):
    dropout_pr = 0.0
    heads: int = 4
    attn_dim: int = 128
    hidden_dim: int = 128
    target_size: int = 20
    sl_alpha: float = 2.0

    # In prod, BF16 is used by HSTU attention
    dtype = torch.bfloat16
    invalid_attn_mask_type = "lower_triangular"
    causal = True
    alpha = 1.0 / attn_dim * 10000

    # generate inputs
    torch.manual_seed(1001)  # for reproducibility
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_SL(lengths, sl_alpha, max_seq_len=max_seq_len)
    num_targets = torch.randint(
        1,
        target_size + 1,
        (batch_size,),
        device=lengths.device,
        dtype=lengths.dtype,
    )
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

    q = _switch_to_contiguous_if_needed(q)
    k = _switch_to_contiguous_if_needed(k)
    v = _switch_to_contiguous_if_needed(v)

    sanity_check_attention(
        max_seq_len=max_seq_len,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type=invalid_attn_mask_type,
        dropout_pr=dropout_pr,
        attn_bias=None,
        max_attn_len=None,
        contextual_seq_len=0,
    )

    fn = lambda: _AttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        0,  # max_attn_len,
        0,  # contextual_seq_len
        True,  # sort_by_length,
    )

    fn_ref = lambda: torch_hstu_attention(
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
        max_attn_len = 0,
        contextual_seq_len=0,
        min_full_attn_seq_len=0,
    )

    out = fn() * max_seq_len
    out_ref = fn_ref() * max_seq_len
    print(f"out = {out}")
    print(f"out_ref = {out_ref}")
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=0)

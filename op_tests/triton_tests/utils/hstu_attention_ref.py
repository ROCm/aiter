import torch
from typing import Optional, Tuple
import torch.nn.functional as F


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
def jagged_to_padded_dense(
    q: torch.Tensor, offsets: torch.Tensor, max_seq_len: int, padding_value
):
    assert len(q.shape) == 2, "q needs to be 2-dim tensor"
    L, D = q.shape
    B = offsets.shape[0] - 1
    padded_shape = (B, max_seq_len, D)
    padded_q = torch.full(padded_shape, padding_value, dtype=q.dtype, device=q.device)
    for i in range(B):
        s = offsets[i]
        e = offsets[i + 1]
        padded_q[i][0 : e - s] = q[s:e]

    return padded_q


# pad sequence according to max sequence len
def pad_sequence(q: torch.Tensor, seq_offsets: torch.Tensor, N: int, padding_value):
    L, D = q.shape
    padded_q = jagged_to_padded_dense(
        q.reshape(L, D), offsets=seq_offsets, max_seq_len=N, padding_value=0.0
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
    padded_q = (
        pad_sequence(q.reshape(L, H * D), seq_offsets, N, 0.0)
        .view(-1, N, H, D)
        .transpose(1, 2)
    )
    padded_k = (
        pad_sequence(k.reshape(L, H * D), seq_offsets, N, 0.0)
        .view(-1, N, H, D)
        .transpose(1, 2)
    )
    padded_v = (
        pad_sequence(v.reshape(L, H * D), seq_offsets, N, 0.0)
        .view(-1, N, H, D)
        .transpose(1, 2)
    )

    return padded_q, padded_k, padded_v


# convert sequences from dense format to jagged format
def dense_to_jagged(seq: torch.Tensor, offsets: torch.Tensor, L: int):
    B, N, HV = seq.shape
    assert L == offsets[-1], f"jagged dim mismatch {offsets[-1]} != {L}!"
    out = torch.empty((L, HV), dtype=seq.dtype, device=seq.device)

    for i in range(B):
        s = offsets[i]
        e = offsets[i + 1]
        out[s:e] = seq[i][0 : e - s]

    return out


# torch hstu reference implementation
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

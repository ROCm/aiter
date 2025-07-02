
import fbgemm_gpu
import torch
from typing import List, Optional, Tuple
import torch.nn.functional as F

# """
# Timestamp bias
# """
# def get_time_bias_bias(
#     ts: torch.Tensor,
#     ts_weights: torch.Tensor,
#     causal: bool,
#     bucket_function: str,
#     bucket_incr: float,
#     final_div: float,
#     delta: float,
#     num_buckets: int,
#     N: int,) -> torch.Tensor:

#     if causal:
#         ts = ts[:, 1:].unsqueeze(2) - ts[:, :-1].unsqueeze(1)
#     else:
#         ts = ts[:, :-1].unsqueeze(2) - ts[:, 1:].unsqueeze(1)

#     ts = ts.view(-1)
#     ts = ts + delta
#     ts = ts.clamp(min=1e-6) / bucket_incr
#     if bucket_function == "sqrt":
#         ts = torch.sqrt(ts)
#     ts = (ts / final_div).clamp(min=0).int

#     ts = torch.clamp(
#         ts,
#         min=0,
#         max=num_buckets,)
#     ts_weights = torch.index_select(ts_weights.view(-1), index=ts.view(-1), dim=0)
#     ts_weights = ts_weights.view(-1, N, N)

#     return ts_weights


# """
# Position bias
# """
# def pos_bias_fn(
#     pos_weights: torch.Tensor,
#     N: int,
#     seq_offsets: torch.Tensor,
#     invalid_attn_mask_type: str,
#     num_targets: Optional[torch.Tensor] = None,
#     max_pos_ind: Optional[int] = None,
# ) -> torch.Tensor:
#     ids = torch.arange(0, N, device=pos_weights.device)
#     row_ids = ids.view(N, 1).expand(N, N)
#     col_ids = ids.expand(N, N)
#     if num_targets is not None:
#         seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
#         seq_lengths = seq_lengths.view(-1, 1)
#         num_targets = num_targets.view(-1, 1)
#         if invalid_attn_mask_type == "lower_triangular":
#             ids = torch.clamp(ids.view(1, N), max=seq_lengths - num_targets)
#         else:
#             ids = torch.clamp(ids.view(1, N), min=num_targets - 1)
#         row_ids = ids.view(-1, N, 1).expand(-1, N, N)
#         col_ids = ids.view(-1, 1, N).expand(-1, N, N)
#     else:
#         row_ids = ids.view(N, 1).expand(N, N)
#         col_ids = row_ids.t()
#     pos_ids = col_ids - row_ids

#     if max_pos_ind is not None:
#         pos_ids = pos_ids + max_pos_ind - 1
#         pos_ids = torch.clamp(pos_ids, min=0, max=2 * max_pos_ind - 2)
#     else:
#         pos_ids = pos_ids + N - 1
#     pos_weight = torch.index_select(pos_weights, 0, pos_ids.view(-1))
#     pos_weight = pos_weight.view(-1, N, N)

#     return pos_weight


# def get_invalid_mask(N: int, max_attn_len: int, causal: bool):
#     seq= torch.arange(N).cuda()
#     seq_m = torch.unsqueeze(seq, 1)
#     seq_n = torch.unsqueeze(seq, 0)
#     offs_m_minus_n = seq_m - seq_n
#     invalid_mask = seq_m >= seq_n
#     # if max_attn_len > 0:
#     #     invalid_mask = invalid_mask and (offs_m_minus_n <= max_attn_len)

#     return invalid_mask


# def torch_hstu_attention_fwd(
#     N: int,
#     alpha: float,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     seq_offsets: torch.Tensor,
#     causal: bool,
#     num_targets: Optional[torch.Tensor],
#     max_attn_len: int, # None
#     contextual_seq_len: int, # false
#     sort_by_length_indices: Optional[torch.Tensor],
# ) -> torch.Tensor:
#     seq_start = seq_offsets[0:-1]
#     seq_end = seq_offsets[1:]
#     seq_len = seq_end - seq_start

#     out = torch.empty_like(v)
#     for i in range(len(seq_start)):
#         ss = seq_offsets[i]
#         se = seq_offsets[i + 1]
#         ss_len = se - ss
#         mask = get_invalid_mask(ss_len, max_attn_len, causal)
#         in_q = q[ss : se].permute(1, 0, 2)
#         in_k = k[ss : se].permute(1, 0, 2)
#         in_v = v[ss : se].permute(1, 0, 2)
#         n_targets = num_targets[i] if num_targets is not None else 0
#         end = ss_len - n_targets if causal else ss_len

#         if end < ss_len:
#             pad_shape = in_q.shape
#             pad_shape1 = (pad_shape[0], ss_len - end, pad_shape[2])
#             padding_0 = torch.zeros(pad_shape1, dtype=in_q.dtype, device=in_q.device)
#             tmp_q = in_q[end : ss_len]
#             in_q[:, end : ss_len, :] = padding_0
#             in_k[:, end : ss_len, :] = padding_0
#             in_v[:, end : ss_len, :] = padding_0

#         qk = torch.bmm(in_q, in_k.permute(0, 2, 1))
#         silu = qk / (torch.exp(-qk) + 1.0) / N
#         silu = silu * mask
#         qkv = torch.bmm(silu, in_v)
#         out[ss : se] = qkv.permute(1, 0, 2)
    
#     return out

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


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]

    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


def pytorch_hstu_mha(
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
    q, k, v = _pad_qkv(
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
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


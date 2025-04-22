import torch
from typing import List, Optional, Tuple

"""
Timestamp bias
"""
def get_time_bias_bias(
    ts: torch.Tensor,
    ts_weights: torch.Tensor,
    causal: bool,
    bucket_function: str,
    bucket_incr: float,
    final_div: float,
    delta: float,
    num_buckets: int,
    N: int,) -> torch.Tensor:

    if causal:
        ts = ts[:, 1:].unsqueeze(2) - ts[:, :-1].unsqueeze(1)
    else:
        ts = ts[:, :-1].unsqueeze(2) - ts[:, 1:].unsqueeze(1)

    ts = ts.view(-1)
    ts = ts + delta
    ts = ts.clamp(min=1e-6) / bucket_incr
    if bucket_function == "sqrt":
        ts = torch.sqrt(ts)
    ts = (ts / final_div).clamp(min=0).int

    ts = torch.clamp(
        ts,
        min=0,
        max=num_buckets,)
    ts_weights = torch.index_select(ts_weights.view(-1), index=ts.view(-1), dim=0)
    ts_weights = ts_weights.view(-1, N, N)

    return ts_weights


"""
Position bias
"""
def pos_bias_fn(
    pos_weights: torch.Tensor,
    N: int,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor] = None,
    max_pos_ind: Optional[int] = None,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=pos_weights.device)
    row_ids = ids.view(N, 1).expand(N, N)
    col_ids = ids.expand(N, N)
    if num_targets is not None:
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        seq_lengths = seq_lengths.view(-1, 1)
        num_targets = num_targets.view(-1, 1)
        if invalid_attn_mask_type == "lower_triangular":
            ids = torch.clamp(ids.view(1, N), max=seq_lengths - num_targets)
        else:
            ids = torch.clamp(ids.view(1, N), min=num_targets - 1)
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
    pos_ids = col_ids - row_ids

    if max_pos_ind is not None:
        pos_ids = pos_ids + max_pos_ind - 1
        pos_ids = torch.clamp(pos_ids, min=0, max=2 * max_pos_ind - 2)
    else:
        pos_ids = pos_ids + N - 1
    pos_weight = torch.index_select(pos_weights, 0, pos_ids.view(-1))
    pos_weight = pos_weight.view(-1, N, N)

    return pos_weight


def get_invalid_mask(N: int, max_attn_len: int, causal: bool):
    seq= torch.arange(N)
    seq_m = torch.unsqueeze(seq, 1)
    seq_n = torch.unsqueeze(seq, 0)
    offs_m_minus_n = seq_m - seq_n
    invalid_mask = seq_m >= seq_n
    if max_attn_len is not None:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len

    return invalid_mask


def torch_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int, # None
    contextual_seq_len: int, # false
    sort_by_length_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    seq_start = seq_offsets[0:-1]
    seq_end = seq_offsets[1:]
    seq_len = seq_end - seq_start

    out = torch.empty_like(v)
    mask = get_invalid_mask(N, max_attn_len, causal)
    for i in range(len(seq_start)):
        in_q = q[seq_start:seq_end].permute(1, 0, 2)
        in_k = k[seq_start:seq_end].permute(1, 0, 2)
        in_v = v[seq_start:seq_end].permute(1, 0, 2)
        n_targets = num_targets[i] if num_targets is not None else 0
        end = seq_len - n_targets if causal else seq_len
        in_q[end:] = 0
        in_k[end:] = 0
        in_v[end:] = 0

        qk = torch.bmm(in_q, in_k)
        silu = qk / (torch.exp(-qk) + 1.0) / N
        silu = silu * mask
        qkv = torch.bmm(silu, in_v)
        out[seq_start:seq_end] = qkv.permute(1, 0, 2)
    
    return out


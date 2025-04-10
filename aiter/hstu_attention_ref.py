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

def 
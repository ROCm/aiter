"""DSV4 sparse_attn ABI validator (issue sunway513/atom#37).

Host-side checker that translates GPU OOB errors into readable
ValueError messages. Enable via ATOM_AITER_VALIDATE=1 in ATOM; default
off in prod for zero overhead.
"""

from __future__ import annotations

import torch


def dsv4_validate_sparse_attn_metadata(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    pool_capacity: int,
) -> None:
    """Raises ValueError with the first failed constraint.

    See sunway513/ATOM docs/superpowers/specs/2026-04-26-dsv4-w43-redo-aiter-track-design.md
    §5 for the full ABI contract.
    """
    # ---- 1. Tensor shape & rank --------------------------------------
    if q.dim() != 4:
        raise ValueError(f"q must be 4-D [B,M,H,D], got {tuple(q.shape)}")
    if kv.dim() != 3:
        raise ValueError(f"kv must be 3-D [B,N,D], got {tuple(kv.shape)}")
    if q.shape[0] != kv.shape[0]:
        raise ValueError(f"q.B={q.shape[0]} != kv.B={kv.shape[0]}")
    if q.shape[-1] != kv.shape[-1]:
        raise ValueError(f"q.head_dim={q.shape[-1]} != kv.head_dim={kv.shape[-1]}")
    if topk_idxs.dim() != 3:
        raise ValueError(f"topk_idxs must be 3-D [B,M,K], got {tuple(topk_idxs.shape)}")
    if topk_idxs.shape[0] != q.shape[0] or topk_idxs.shape[1] != q.shape[1]:
        raise ValueError(
            f"topk_idxs.shape[:2]={tuple(topk_idxs.shape[:2])} != "
            f"q.shape[:2]={tuple(q.shape[:2])}"
        )

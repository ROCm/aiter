# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compact or ragged-``next_n`` batch for paged FP8 MQA logits.

Q is dense ``[B, next_n, H, D]``. For a uniform batch all rows are live, like
Gluon. For a ragged batch, ``next_n_lens[b]`` in ``{1..next_n}`` marks how many
leading rows are live. The gfx950 FlyDSL kernel consumes that optional vector;
unused Q/weight rows are ignored and unused logit rows stay ``-inf``.
"""

from __future__ import annotations

import torch

from op_tests.flydsl_tests.test_flydsl_fp8_paged_mqa_logits import (
    ref_fp8_paged_mqa_logits,
)

MAX_NN = 8


def sample_next_n_lens(nq: int, max_nn: int = MAX_NN, seed: int = 1079) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + nq)
    return torch.randint(
        1,
        max_nn + 1,
        (nq,),
        generator=g,
        dtype=torch.int32,
        device="cpu",
    )


def padded_logits(batch: int, max_nn: int, max_model_len: int) -> torch.Tensor:
    return torch.full(
        (batch * max_nn, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )


def scatter_seq_logits(
    out: torch.Tensor,
    seq: int,
    max_nn: int,
    row_logits: torch.Tensor,
    n_rows: int,
) -> None:
    """Copy ``n_rows`` of a compact ``[n, max_model_len]`` tensor into padded out."""
    dst = seq * max_nn
    out[dst : dst + n_rows].copy_(row_logits[:n_rows])


def ref_padded_ragged(
    q_pad: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights_pad: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    next_n_lens: torch.Tensor,
    max_model_len: int,
    fp8_dtype,
    *,
    max_nn: int = MAX_NN,
    block_size: int = 64,
) -> torch.Tensor:
    """Per-sequence compact ref, scattered into ``[B*max_nn, max_model_len]``."""
    batch = q_pad.shape[0]
    out = padded_logits(batch, max_nn, max_model_len)
    nn = next_n_lens.tolist()
    for b, n in enumerate(nn):
        n = int(n)
        q_b = q_pad[b : b + 1, :n]
        w_b = weights_pad[b * max_nn : b * max_nn + n]
        compact = ref_fp8_paged_mqa_logits(
            q_b,
            kv_cache_fp8,
            w_b,
            context_lens[b : b + 1],
            block_tables[b : b + 1],
            max_model_len,
            fp8_dtype,
            block_size=block_size,
        )
        scatter_seq_logits(out, b, max_nn, compact, n)
    return out


def live_row_mask(
    next_n_lens: torch.Tensor, max_nn: int, max_model_len: int
) -> torch.Tensor:
    """True on live ``[B*max_nn, max_model_len]`` rows (all columns)."""
    batch = next_n_lens.shape[0]
    rows = torch.arange(max_nn, device=next_n_lens.device)
    live = rows[None, :] < next_n_lens[:, None]
    return live.reshape(batch * max_nn, 1).expand(-1, max_model_len)

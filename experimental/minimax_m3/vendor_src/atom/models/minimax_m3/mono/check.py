# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layer-by-layer comparison of the mono path against the original modules.

Debug only (``ATOM_MONO_CHECK=1``, meant for ``--enforce-eager``). The dense
layers run twice, mono then the original modules, compared where they hand the
first sparse layer its (ar, res); the step goes on from mono's. Every sparse
layer runs twice on the same (ar, res): first the mono layer kernel, then the
original decoder layer, fed ``ar`` on rank 0 and zeros elsewhere so its fused
all-reduce reproduces ``ar`` exactly. Both write the same cache bytes for the
step's tokens (K1 is bit-exact with the original insert), so running both is
side-effect free; mono going first reads only the cache entries it inserted.
Rank 0 logs, per layer and token, the difference of the residual and of the
reduced output.
"""

import json
import logging

import torch
from aiter import topk_select

from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.models.minimax_m3.mono.config import HIDDEN
from atom.models.minimax_m3.mono.runner import N_DENSE, token_rows
from atom.mono.runtime.compare import compare
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def forward_checked(
    runner, input_ids: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    fwd = get_forward_context()
    runner.mailboxes.begin_step()
    ars = runner.ars
    # the dense layers, mono first, then the original ones on the same step,
    # compared where they hand the first sparse layer its (ar, res)
    res = runner.run_dense_mono(input_ids, positions, fwd, (), []).clone()
    n = res.shape[0]
    ar = ars[0][:n].clone()
    ref_res = runner.run_dense(input_ids, positions)
    if runner.rank == 0:
        for k in range(n):
            logger.info(
                "mono check dense layers token %d/%d pos %d: residual %s | "
                "reduced out %s",
                k,
                n,
                int(positions[k]),
                compare(res[k], ref_res[k]),
                compare(ar[k], ars[0][k]),
            )
    ars[0][:n].copy_(ar)
    rows = token_rows(fwd, n)
    zeros = torch.zeros(n, HIDDEN, dtype=ars[0].dtype, device=ars[0].device)
    for i, lw in enumerate(runner.sparse):
        layer = runner.model.layers[lw.layer_id]
        h_in = ars[i % 2][:n].clone() if runner.rank == 0 else zeros
        res_in = res.clone()
        # mono first: it reads only the caches it inserts itself, as in a real step
        res = runner.run_sparse_layer(i, lw, fwd, positions, res, rows)
        ref_partial, ref_res = layer(positions, h_in, res_in)
        ref_ar = tensor_model_parallel_all_reduce(ref_partial).view(n, HIDDEN)
        if runner.rank == 0:
            for k in range(n):
                logger.info(
                    "mono check layer %d token %d/%d pos %d seq_len %d: residual %s | "
                    "reduced out %s",
                    lw.layer_id,
                    k,
                    n,
                    int(positions[k]),
                    int(rows[1][k]),
                    compare(res[k], ref_res.view(n, HIDDEN)[k]),
                    compare(ars[(i + 1) % 2][k], ref_ar[k]),
                )
    return runner.finish_step(res)


def _fingerprint(t: torch.Tensor) -> list[int]:
    """Two integer sums of a tensor's 16-bit words (plain and index-weighted):
    equal fingerprints mean equal bits for any difference a run could make."""
    w = t.contiguous().view(torch.int16).flatten().to(torch.int64)
    idx = torch.arange(1, w.numel() + 1, device=w.device)
    return [int(w.sum()), int((w * idx).sum())]


def forward_probed(
    runner, input_ids: torch.Tensor, positions: torch.Tensor, sums: dict
) -> torch.Tensor:
    """``runner.forward`` without aux layers, filling ``sums`` with the fingerprint
    of every stage: each dense layer's (hidden, residual), the first sparse layer's
    ``ar``, each sparse layer's (ar, h_mid) and the final hidden."""
    n = input_ids.numel()
    model = runner.model
    runner.mailboxes.begin_step()
    hidden = model.get_input_embeddings(input_ids)
    sums["embed"] = _fingerprint(hidden)
    residual = None
    for i in range(N_DENSE):
        hidden, residual = model.layers[i](positions, hidden, residual)
        sums[f"dense{i}"] = _fingerprint(hidden) + _fingerprint(residual)
    runner.ars[0][:n].copy_(tensor_model_parallel_all_reduce(hidden).view(n, HIDDEN))
    sums["ar_in"] = _fingerprint(runner.ars[0][:n])
    res = residual.view(n, HIDDEN)
    fwd = get_forward_context()
    rows = token_rows(fwd, n)
    # token 0's full blocks (the tail one also holds stale slots past the token)
    blocks = rows[0][0][: (int(rows[1][0]) - 1) // 128].long()
    for i, lw in enumerate(runner.sparse):
        # the history this layer reads, before this step inserts
        kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
        sums[f"cache{lw.layer_id}"] = (
            _fingerprint(kv.k_cache[blocks])
            + _fingerprint(kv.v_cache[blocks])
            + _fingerprint(lw.attn_impl.index_cache[blocks])
        )
        res = runner.run_sparse_layer(i, lw, fwd, positions, res, rows)
        sums[f"L{lw.layer_id}"] = _fingerprint(
            runner.ars[(i + 1) % 2][:n]
        ) + _fingerprint(res)
    out = runner.finish_step(res)
    sums["final"] = _fingerprint(out)
    return out


def trace_logits(
    causal_lm,
    rank: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden,
    path,
    sums: dict | None = None,
) -> None:
    """Append this step's rows to ``path`` (rank 0, one JSON line per step): each
    row's input token, position and top-2 logits, to align two runs of one prompt
    token by token. ``hidden``: the mono forward's output."""
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    raw = causal_lm.compute_logits(hidden)
    pick = topk_select(raw, 1, tie="low")[1].view(-1)
    vals, ids = raw.float().topk(2, dim=-1)
    if rank == 0:
        with open(path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ids": input_ids.tolist(),
                        "pos": positions.tolist(),
                        "top_ids": ids.tolist(),
                        "top_vals": vals.tolist(),
                        "pick": pick.tolist(),
                        "sums": sums,
                    }
                )
                + "\n"
            )

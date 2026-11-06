# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Top-k / top-p sampling from bf16 logits in two kernels.

This is the draw of the top-k-first sampler (``top_k_top_p_sampling_from_probs``
with ``AITER_TOPK_FIRST_SMALL_K_TOPP=1``) applied to
``softmax(logits / t)``, without the fp32 probabilities in memory. The
logits are read once. Only the k largest columns of each row get a
probability.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.topk_topp_sample import (
    GROUP,
    _topk_topp_sample_select_kernel,
    _topk_topp_sample_stats_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = [
    "MAX_K",
    "topk_topp_sample",
    "topk_topp_sample_gluon_select_supported",
    "topk_topp_sample_supported",
]

MAX_K = 64
# the Gluon select packs a group index in 13 bits and a chunk index in 10 bits
_GLUON_SELECT_MAX_VOCAB = 8192 * 32


def _get_config(rows: int) -> tuple[dict, dict]:
    config_dir = resolve_config_dir("fusions", "TOPK_TOPP_SAMPLE")
    configs = load_config_json(f"{config_dir}/DEFAULT.json", required=True)
    stats = select_leq_config(configs["stats"], rows, prefix="ROWS_LEQ_")
    return stats, dict(configs["select"])


def _get_gluon_select_config(rows: int) -> dict:
    config_dir = resolve_config_dir("fusions", "TOPK_TOPP_SAMPLE", backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json", required=True)
    return select_leq_config(configs, rows, prefix="ROWS_LEQ_")


def topk_topp_sample_supported(logits: torch.Tensor, top_k) -> bool:
    """True when ``topk_topp_sample`` takes these logits and this top_k."""
    return (
        logits.is_cuda
        and logits.dim() == 2
        and logits.dtype == torch.bfloat16
        and logits.stride(1) == 1
        and isinstance(top_k, int)
        and 0 < top_k <= MAX_K
        and top_k * GROUP.value <= logits.shape[1]
    )


def topk_topp_sample_gluon_select_supported(logits: torch.Tensor, top_k) -> bool:
    """True when ``topk_topp_sample(..., gluon_select=True)`` takes these logits."""
    return (
        topk_topp_sample_supported(logits, top_k)
        and get_arch() == "gfx1250"
        and logits.shape[1] <= _GLUON_SELECT_MAX_VOCAB
    )


def topk_topp_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    top_p: float | torch.Tensor,
    *,
    verify_rows: torch.Tensor | None = None,
    bonus_rows: torch.Tensor | None = None,
    cu_num_draft_tokens: torch.Tensor | None = None,
    uniform: torch.Tensor | None = None,
    eps: float = 1e-10,
    out: torch.Tensor | None = None,
    debug: torch.Tensor | None = None,
    presence: dict | None = None,
    gluon_select: bool = False,
    philox: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Sample one token a row from ``softmax(logits / t)`` under top-k and top-p.

    The result is the draw of the top-k-first sampler: the k largest
    probabilities, the top-p set on their full-vocab mass (an element stays
    when the mass above it is below p), and a draw from the renormalised set
    with one uniform a row. The k columns are the largest logits, the lowest
    column first on a tie, in descending order. A row with ``t <= eps`` takes
    the argmax (the lowest column on a tie).

    Args:
        logits: [R, V] bf16, dense along V.
        temperatures: fp32, one value a request.
        top_k: number of columns, at most ``MAX_K``.
        top_p: float, or fp32 tensor with one value a request.
        verify_rows, bonus_rows: int tensors. When given, output row r reads
            logits row ``verify_rows[r]`` (r < len(verify_rows)), then
            ``bonus_rows[r - len(verify_rows)]``. Request of a verify row:
            ``searchsorted(cu_num_draft_tokens, r, right=True)``; of a bonus
            row: ``r - len(verify_rows)``. When None, row r is request r.
        uniform: fp32 [n] U[0, 1) draws. When None: ``torch.rand(n)``, the
            draw that ``top_p_sampling_small_k`` makes.
        out: int32 [n] result buffer.
        debug: int32 [n, 3 * KB + 2] buffer for the k columns, their
            renormalised and softmax probabilities, the top-p threshold and the
            softmax sum (tests).
        presence: a presence penalty applied to the logits before everything
            else, as ``bf16(x - p)`` on each penalized column. Keys, indexed by
            the logits row R or its request s: ``input_ids``, ``row_seq`` [R],
            ``row_pos`` [R], ``seq_start`` [s], ``slots`` [s], ``penalties``
            [s] fp32, ``mask`` uint8 [slots, V] and ``max_extra``. A column is
            penalized when ``mask[slots[s], col]`` is set, or when it is one of
            ``input_ids[seq_start[s] + 1 .. seq_start[s] + row_pos[R]]`` (the
            drafts before a verify row). Optional key ``anchor``
            (gluon_select only): also penalize ``input_ids[seq_start[s]]``,
            the query's first token, and set its byte in the mask.
        gluon_select: the select pass as a Gluon kernel with fewer dependent
            steps (gfx1250, see ``topk_topp_sample_gluon_select_supported``).
            The same tokens as the Triton select kernel.
        philox: ``(seed, offset)``, gluon_select only: the uniform of row r is
            Philox-4x32-10 at key ``seed`` and counter ``(r, offset)``, made
            in the kernel, in place of ``uniform``. Use a new offset for
            each call.

    Returns:
        int32 [n] token ids.
    """
    assert topk_topp_sample_supported(logits, top_k), "unsupported logits or top_k"
    R, V = logits.shape
    device = logits.device
    row_map = verify_rows is not None
    if row_map:
        assert bonus_rows is not None and cu_num_draft_tokens is not None
        n_verify, n_cu = verify_rows.numel(), cu_num_draft_tokens.numel()
        n = n_verify + bonus_rows.numel()
    else:
        verify_rows = bonus_rows = cu_num_draft_tokens = temperatures
        n_verify = n_cu = 0
        n = R
    if out is None:
        out = torch.empty(n, dtype=torch.int32, device=device)
    if n == 0:
        return out
    anchor = presence is not None and bool(presence.get("anchor", False))
    assert gluon_select or (
        philox is None and not anchor
    ), "philox and anchor need gluon_select"
    assert philox is None or uniform is None, "philox replaces uniform"
    if uniform is None and philox is None:
        uniform = torch.rand(n, dtype=torch.float32, device=device)
    _LOGGER.info(
        "TOPK_TOPP_SAMPLE: logits=%s rows=%d k=%d", tuple(logits.shape), n, top_k
    )

    stats_cfg, select_cfg = _get_config(n)
    groups = stats_cfg["GROUPS"]
    NG = triton.cdiv(V, GROUP.value)
    NCH = triton.cdiv(NG, groups)
    NG_P2 = triton.next_power_of_2(NG)
    KB = max(16, triton.next_power_of_2(top_k))
    block_bs = max(16, triton.next_power_of_2(max(n_cu, 1)))
    m = torch.empty((n, NCH), dtype=torch.float32, device=device)
    z = torch.empty((n, NCH), dtype=torch.float32, device=device)
    g = torch.empty((n, NG), dtype=torch.int16, device=device)
    penalty = presence is not None
    if penalty:
        pen = (
            presence["input_ids"],
            presence["row_seq"],
            presence["row_pos"],
            presence["seq_start"],
            presence["slots"],
            presence["penalties"],
            presence["mask"],
            presence["mask"].stride(0),
        )
        max_extra = max(int(presence["max_extra"]), 1)
    else:
        pen = (out,) * 7 + (0,)
        max_extra = 1
    per_req = isinstance(top_p, torch.Tensor)
    top_p_t = top_p.float().contiguous() if per_req else temperatures

    if gluon_select:
        assert topk_topp_sample_gluon_select_supported(logits, top_k), "gfx1250 only"
        from aiter.ops.triton._gluon_kernels.gfx1250.fusions.topk_topp_sample import (
            _topk_topp_sample_select_gluon_kernel,
            _topk_topp_sample_stats_gluon_kernel,
        )

        gcfg = _get_gluon_select_config(n)
        # 4 waves a chunk: the Triton stats kernel's reduction order
        gcpp = gcfg["stats_chunks"]
        _topk_topp_sample_stats_gluon_kernel[(n, triton.cdiv(NCH, gcpp))](
            logits,
            logits.stride(0),
            verify_rows,
            bonus_rows,
            cu_num_draft_tokens,
            temperatures,
            m,
            z,
            g,
            *pen,
            n_verify,
            n_cu,
            V,
            NG,
            NCH,
            ROW_MAP=row_map,
            BLOCK_BS=block_bs,
            PENALTY=penalty,
            MAX_EXTRA=max_extra,
            ANCHOR=anchor,
            GROUPS=groups,
            CPP=gcpp,
            NW=4 * gcpp,
            num_warps=4 * gcpp,
        )
        nw = gcfg["num_warps"]
        seed, offset = philox if philox is not None else (0, 0)
        _topk_topp_sample_select_gluon_kernel[(n,)](
            logits,
            logits.stride(0),
            verify_rows,
            bonus_rows,
            cu_num_draft_tokens,
            temperatures,
            top_p_t,
            0.0 if per_req else float(top_p),
            uniform if uniform is not None else out,
            seed & 0xFFFFFFFF,
            (seed >> 32) & 0xFFFFFFFF,
            offset & 0xFFFFFFFF,
            (offset >> 32) & 0xFFFFFFFF,
            m,
            z,
            g,
            out,
            debug if debug is not None else out,
            *pen,
            presence["mask"].shape[1] if penalty else 0,
            n_verify,
            n_cu,
            V,
            NG,
            NCH,
            top_k,
            eps,
            ROW_MAP=row_map,
            BLOCK_BS=block_bs,
            PENALTY=penalty,
            MAX_EXTRA=max_extra,
            ANCHOR=anchor,
            GROUPS=groups,
            NCH_P2=triton.next_power_of_2(NCH),
            KB=KB,
            CAP=2 * KB,
            TOP_P_PER_REQ=per_req,
            PHILOX=philox is not None,
            DEBUG=debug is not None,
            NW=nw,
            num_warps=nw,
        )
        return out
    cpp = stats_cfg["CHUNKS_PER_PROGRAM"]
    _topk_topp_sample_stats_kernel[(n, triton.cdiv(NCH, cpp))](
        logits,
        logits.stride(0),
        verify_rows,
        bonus_rows,
        cu_num_draft_tokens,
        temperatures,
        m,
        z,
        g,
        *pen,
        n_verify,
        n_cu,
        V,
        NG,
        NCH,
        ROW_MAP=row_map,
        BLOCK_BS=block_bs,
        PENALTY=penalty,
        MAX_EXTRA=max_extra,
        GROUPS=groups,
        CHUNKS_PER_PROGRAM=cpp,
        num_warps=stats_cfg["num_warps"],
    )
    ws = torch.empty((n, NG_P2 + 2 * KB), dtype=torch.int32, device=device)
    _topk_topp_sample_select_kernel[(n,)](
        logits,
        logits.stride(0),
        verify_rows,
        bonus_rows,
        cu_num_draft_tokens,
        temperatures,
        top_p_t,
        0.0 if per_req else float(top_p),
        uniform,
        m,
        z,
        g,
        ws,
        out,
        debug if debug is not None else out,
        *pen,
        n_verify,
        n_cu,
        V,
        NG,
        NCH,
        top_k,
        eps,
        ROW_MAP=row_map,
        BLOCK_BS=block_bs,
        NCH_P2=triton.next_power_of_2(NCH),
        NG_P2=NG_P2,
        NB=max(32, KB),
        CAPG=select_cfg["CAPG"],
        KB=KB,
        TOP_P_PER_REQ=per_req,
        DEBUG=debug is not None,
        PENALTY=penalty,
        MAX_EXTRA=max_extra,
        num_warps=select_cfg["num_warps"],
    )
    return out

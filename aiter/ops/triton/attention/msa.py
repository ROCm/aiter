# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax sparse attention (MSA) over a k4v8 KV cache.

MSA (MiniMax-M3) is grouped-query attention where each (query row, KV head)
attends only the top-k 128-token blocks its indexer selected. One sparse
block is one KV page.

k4v8 cache: K in MXFP4 (E8M0 scale per 32 dims of the QK dot dim, preshuffled
for the scaled MFMA), V in fp8 e4m3 with one per-tensor scale. The record
layout is documented in the kernel module
(aiter.ops.triton._triton_kernels.attention.msa.py).

    kv_cache = msa_k4v8_alloc(num_pages, num_kv_heads, device)
    msa_k4v8_store(k, v, kv_cache, slot_mapping, v_scale)
    out = msa_fwd(q, kv_cache, topk, block_table, sm_scale, v_scale,
                  seq_lens=seq_lens, decode_query_len=8)          # decode
    out = msa_fwd(q, kv_cache, topk, block_table, sm_scale, v_scale,
                  query_req=req, query_pos=pos)                    # prefill
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.msa import (
    HEAD_DIM,
    K_BYTES,
    KS_BYTES,
    P_SCALE,
    PAGE,
    RECORD_BYTES,
    V_BYTES,
    msa_gfx950_triton_kernel,
    msa_k4v8_store_kernel,
    msa_merge_kernel,
    msa_q_mxfp8_kernel,
    msa_union_kernel,
    RECORD4_BYTES,
    V4_BYTES,
    V4S_BYTES,
    msa_k4v4_store_kernel,
)

__all__ = [
    "PAGE",
    "HEAD_DIM",
    "RECORD_BYTES",
    "msa_k4v8_alloc",
    "msa_k4v8_store",
    "msa_k4v8_dequant",
    "msa_fwd",
    "msa_decode_splits",
    "msa_workspace",
]

_FP8 = torch.float8_e4m3fn  # gfx950 fp8 is OCP e4m3
_Q_MODES = {"bf16": 0, "mxfp8": 1, "mxfp8x2": 2}


def msa_k4v8_alloc(num_pages: int, num_kv_heads: int, device) -> torch.Tensor:
    """A zeroed k4v8 cache, uint8 [pages, kv_heads, RECORD_BYTES]."""
    return torch.zeros(
        (num_pages, num_kv_heads, RECORD_BYTES), dtype=torch.uint8, device=device
    )


def _check_cache(kv_cache: torch.Tensor) -> None:
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3:
        raise ValueError("kv_cache must be uint8 [pages, kv_heads, RECORD_BYTES]")
    if kv_cache.shape[2] != RECORD_BYTES or kv_cache.stride(2) != 1:
        raise ValueError(f"kv_cache records must be {RECORD_BYTES} contiguous bytes")


def _scale_tensor(v_scale, device) -> torch.Tensor:
    if isinstance(v_scale, torch.Tensor):
        return v_scale.reshape(1).to(device=device, dtype=torch.float32)
    return torch.tensor([float(v_scale)], dtype=torch.float32, device=device)


def msa_k4v8_store(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    v_scale: torch.Tensor | float,
) -> None:
    """Quantize k / v ([tokens, kv_heads, 128], bf16/fp16/fp32) into their
    slots (page * PAGE + offset; negative slots are skipped)."""
    _check_cache(kv_cache)
    if k.shape != v.shape or k.ndim != 3 or k.shape[2] != HEAD_DIM:
        raise ValueError("k and v must be [tokens, kv_heads, 128]")
    if k.shape[1] != kv_cache.shape[1]:
        raise ValueError("k has a different KV head count than kv_cache")
    if k.stride(2) != 1 or v.stride(2) != 1:
        raise ValueError("k and v must be contiguous in the head dim")
    n = k.shape[0]
    if n == 0:
        return
    block_t = 16
    vs = _scale_tensor(v_scale, k.device)
    msa_k4v8_store_kernel[(triton.cdiv(n, block_t), k.shape[1])](
        k,
        v,
        slot_mapping,
        kv_cache,
        kv_cache.view(_FP8),
        vs,
        n,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        PAGE=PAGE,
        D=HEAD_DIM,
        K_BYTES=K_BYTES,
        KS_BYTES=KS_BYTES,
        BLOCK_T=block_t,
        num_warps=4,
    )


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def msa_k4v8_dequant(
    kv_cache: torch.Tensor, v_scale: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference unpack: fp32 K and V, each [pages, kv_heads, PAGE, 128]."""
    _check_cache(kv_cache)
    P, H, _ = kv_cache.shape
    raw = kv_cache.reshape(P, H, RECORD_BYTES)
    kb = raw[..., :K_BYTES].reshape(P, H, PAGE // 16, 2, 2, 16, 16)
    kb = kb.permute(0, 1, 2, 5, 3, 4, 6).reshape(P, H, PAGE, HEAD_DIM // 2)
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=kv_cache.device)
    lo, hi = lut[(kb & 0xF).long()], lut[(kb >> 4).long()]
    k = torch.stack([lo, hi], dim=-1).reshape(P, H, PAGE, HEAD_DIM)
    sc = raw[..., K_BYTES : K_BYTES + KS_BYTES].reshape(P, H, PAGE, HEAD_DIM // 32)
    sc = torch.exp2(sc.float() - 127.0)
    k = (k.reshape(P, H, PAGE, HEAD_DIM // 32, 32) * sc[..., None]).reshape(
        P, H, PAGE, HEAD_DIM
    )
    vb = raw[..., K_BYTES + KS_BYTES :].contiguous().view(_FP8).float()
    # [t / 16, d, t % 16] -> [t, d]
    v = vb.reshape(P, H, PAGE // 16, HEAD_DIM, 16).permute(0, 1, 2, 4, 3)
    v = v.reshape(P, H, PAGE, HEAD_DIM) * float(v_scale)
    return k, v


_WORKSPACES: dict = {}


def msa_workspace(device, pairs: int, splits: int, hg: int = 16) -> dict:
    """A workspace for up to `pairs` (row, KV head) pairs x `splits`."""
    n = pairs * splits
    return {
        "part_o": torch.empty((n, hg, HEAD_DIM), dtype=torch.float32, device=device),
        "part_ml": torch.empty((n, hg, 2), dtype=torch.float32, device=device),
        # Zero once; the merging split resets its pair's counter.
        "counter": torch.zeros(pairs, dtype=torch.int32, device=device),
    }


def _workspace(device, pairs, splits, hg, ws):
    if ws is None:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        ws = _WORKSPACES.get(idx)
        if (
            ws is None
            or ws["counter"].numel() < pairs
            or ws["part_ml"].shape[0] < pairs * splits
            or ws["part_o"].shape[1] != hg
        ):
            old = ws
            ws = msa_workspace(
                device,
                max(pairs, old["counter"].numel() if old else 0),
                max(splits, 16),
                hg,
            )
            _WORKSPACES[idx] = ws
    elif ws["counter"].numel() < pairs or ws["part_ml"].shape[0] < pairs * splits:
        raise ValueError("workspace is too small for this call")
    return ws["part_o"], ws["part_ml"], ws["counter"]


def msa_decode_splits(rows: int, num_kv_heads: int, topk: int, cus: int) -> int:
    """Split the top-k blocks until the grid covers about two waves of CUs."""
    work = max(rows * num_kv_heads, 1)
    splits = 1
    while splits < topk and work * splits < 2 * cus:
        splits *= 2
    return splits


_CU_COUNT: dict[int, int] = {}


def _cus(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _CU_COUNT:
        _CU_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _CU_COUNT[idx]


def msa_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    sm_scale: float,
    v_scale: torch.Tensor | float,
    *,
    seq_lens: torch.Tensor | None = None,
    decode_query_len: int | None = None,
    query_req: torch.Tensor | None = None,
    query_pos: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    kv_splits: int | None = None,
    q_mode: str = "mxfp8",
    p_fp8: bool = True,
    workspace: dict | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    waves_per_eu: int = 2,
    backend: str = "triton",
    q_scale: torch.Tensor | None = None,
    block_m: int | None = None,
    kv_format: str = "k4v8",
) -> torch.Tensor:
    """MSA over a k4v8 cache.

    q            [rows, kv_heads * HG, 128] bf16 (HG query heads per KV head)
    kv_cache     uint8 [pages, kv_heads, RECORD_BYTES] (msa_k4v8_store)
    topk         int32 [kv_heads, rows, TOPK] logical block ids, -1 = none
    block_table  int32 [reqs, max_blocks]
    v_scale      the per-tensor V scale the cache was written with

    Decode: seq_lens [reqs] int32 and decode_query_len (rows per request,
    row r is query r % qlen of request r // qlen at seq_len - qlen + r % qlen).
    Prefill: query_req / query_pos, int32 [rows].
    Every row attends the tokens <= its position in its valid blocks; a row
    with none gives 0.

    q_mode: how QK^T meets the MXFP4 K.
      "mxfp8"    Q as one MXFP8 term, one scaled MFMA (default; ~2% rms Q
                 rounding, accuracy-neutral end to end on MiniMax-M3.1)
      "mxfp8x2"  Q as two MXFP8 terms (hi + lo), two scaled MFMAs; about
                 bf16 Q accuracy
      "bf16"     bf16 Q, K upcast to bf16
    p_fp8: P as fp8 (fp8 MFMA against the fp8 V) rather than bf16 (V upcast).
    backend: "triton" (any arch) or "gluon" (gfx950; q_mode mxfp8 / mxfp8x2,
    fp8 P). The gluon backend also takes q already in MXFP8 (e4m3 q
    [rows, H * 128] or [rows, H, 128] with E8M0 q_scale [rows, H * 4], e.g.
    from fused_minimax_m3_norm_rope_quant_insert(out_quant=True)), and
    block_m: rows per program, BLOCK_M = 16 * (query rows), 16..128.
    workspace: the split partials and arrival counters, grown on demand and
    reused across calls (a module-level one per device by default). Size it
    with msa_workspace() before CUDA graph capture.
    """
    if kv_format == "k4v4":
        # K and V MXFP4 (msa_k4v4_store), the per-wave Gluon kernel; v_scale unused
        return _k4v4_fwd(
            q, q_scale, kv_cache, topk, block_table, sm_scale, seq_lens,
            decode_query_len, query_req, query_pos, out, kv_splits, q_mode,
        )  # fmt: skip
    if kv_format != "k4v8":
        raise ValueError(f"kv_format must be 'k4v8' or 'k4v4', got {kv_format!r}")
    _check_cache(kv_cache)
    if backend == "gluon":
        return _gluon_fwd(
            q, q_scale, kv_cache, topk, block_table, sm_scale, v_scale, seq_lens,
            decode_query_len, query_req, query_pos, out, kv_splits, q_mode, p_fp8,
            block_m, num_warps,
        )  # fmt: skip
    rows, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[1]
    if head_dim != HEAD_DIM or num_heads % num_kv_heads:
        raise ValueError("q must be [rows, kv_heads * HG, 128]")
    hg = num_heads // num_kv_heads
    if hg < 16 or hg & (hg - 1):
        raise ValueError("query heads per KV head must be a power of two >= 16")
    if q.dtype != torch.bfloat16 or q.stride(2) != 1:
        raise ValueError("q must be bf16, contiguous in the head dim")
    if topk.dtype != torch.int32 or topk.ndim != 3 or topk.stride(2) != 1:
        raise ValueError("topk must be int32 [kv_heads, rows, TOPK]")
    if topk.shape[0] != num_kv_heads or topk.shape[1] < rows:
        raise ValueError("topk must cover every KV head and row")
    n_topk = topk.shape[2]
    decode = seq_lens is not None
    if decode == (query_req is not None):
        raise ValueError("pass seq_lens + decode_query_len (decode) or query_req + query_pos")
    if decode:
        if not decode_query_len or rows != seq_lens.shape[0] * decode_query_len:
            raise ValueError("rows must be len(seq_lens) * decode_query_len")
        dql = decode_query_len
        query_req = query_pos = seq_lens  # unused
    else:
        if query_pos is None or query_req.shape[0] < rows or query_pos.shape[0] < rows:
            raise ValueError("prefill needs query_req and query_pos for every row")
        dql = 0
        seq_lens = query_req  # unused
    if out is None:
        out = torch.empty_like(q)
    if rows == 0:
        return out
    if kv_splits is None:
        kv_splits = (
            msa_decode_splits(rows, num_kv_heads, n_topk, _cus(q.device)) if decode else 1
        )
    if kv_splits < 1 or kv_splits & (kv_splits - 1) or n_topk % kv_splits:
        raise ValueError("kv_splits must be a power of two dividing TOPK")
    if q_mode not in _Q_MODES:
        raise ValueError(f"q_mode must be one of {tuple(_Q_MODES)}")
    vs = _scale_tensor(v_scale, q.device)
    if kv_splits > 1:
        part_o, part_ml, counter = _workspace(
            q.device, rows * num_kv_heads, kv_splits, hg, workspace
        )
    else:
        part_o = part_ml = counter = out
    if backend != "triton":
        raise ValueError(f"unknown backend {backend!r}")
    msa_gfx950_triton_kernel[(rows, num_kv_heads, kv_splits)](
        q,
        kv_cache,
        kv_cache.view(_FP8),
        topk,
        block_table,
        seq_lens,
        query_req,
        query_pos,
        out,
        part_o,
        part_ml,
        counter,
        vs,
        sm_scale * 1.4426950408889634,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        topk.stride(0),
        topk.stride(1),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        HG=hg,
        D=HEAD_DIM,
        PAGE=PAGE,
        TOPK=n_topk,
        SPLITS=kv_splits,
        DECODE_QL=dql,
        K_BYTES=K_BYTES,
        KS_BYTES=KS_BYTES,
        Q_MODE=_Q_MODES[q_mode],
        P_FP8=p_fp8,
        P_SCALE=P_SCALE,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return out


def msa_gluon_splits(programs: int, topk: int, cus: int, min_pages: int = 4) -> int:
    """Split each program's page union until the grid covers about two waves,
    keeping at least min_pages of a top-k's worth of pages per split (the
    merge costs a [BLOCK_M, 128] fp32 partial per split)."""
    splits = 1
    while splits * min_pages < topk and programs * splits < 2 * cus:
        splits *= 2
    return splits


# k4v8 Gluon K / V loading: registers (default) or async copy into LDS.
_KV_LDS = __import__("os").getenv("AITER_MSA_KV_LDS", "0") == "1"


def _gluon_fwd(
    q, q_scale, kv_cache, topk, block_table, sm_scale, v_scale, seq_lens, dql,
    query_req, query_pos, out, kv_splits, q_mode, p_fp8, block_m, num_warps,
    kv_lds=None,
):  # fmt: skip
    kv_lds = _KV_LDS if kv_lds is None else kv_lds
    from aiter.ops.triton._gluon_kernels.gfx950.attention.msa import (
        msa_gfx950_gluon_kernel,
    )

    if q_mode not in ("mxfp8", "mxfp8x2") or not p_fp8:
        raise ValueError("the gluon backend runs q_mode 'mxfp8' or 'mxfp8x2' with fp8 P")
    rows = q.shape[0]
    num_kv_heads = kv_cache.shape[1]
    decode = seq_lens is not None
    if decode == (query_req is not None):
        raise ValueError("pass seq_lens + decode_query_len (decode) or query_req + query_pos")
    if decode and (not dql or rows != seq_lens.shape[0] * dql):
        raise ValueError("rows must be len(seq_lens) * decode_query_len")
    if topk.dtype != torch.int32 or topk.ndim != 3 or topk.stride(2) != 1:
        raise ValueError("topk must be int32 [kv_heads, rows, TOPK]")
    n_topk = topk.shape[2]
    if q_scale is not None:
        # MXFP8 Q as the insert writes it: one term, unscaled.
        if q_mode != "mxfp8" or q.dtype != _FP8:
            raise ValueError("q_scale takes an e4m3 q with q_mode='mxfp8'")
        num_heads = q.numel() // (rows * HEAD_DIM)
        q8 = q.reshape(rows, num_heads * HEAD_DIM)
        qs = q_scale.reshape(rows, num_heads * HEAD_DIM // 32)
        if q8.stride(1) != 1 or qs.stride(1) != 1:
            raise ValueError("MXFP8 q and q_scale rows must be contiguous")
        if q8.stride(0) != num_heads * HEAD_DIM or qs.stride(0) != num_heads * HEAD_DIM // 32:
            raise ValueError("MXFP8 q and q_scale must be dense [rows, ...]")
        qt = 1
    else:
        if q.dtype != torch.bfloat16 or q.ndim != 3 or q.stride(2) != 1:
            raise ValueError("q must be bf16 [rows, H, 128] (or MXFP8 with q_scale)")
        num_heads = q.shape[1]
        qt = 2 if q_mode == "mxfp8x2" else 1
        q8 = torch.empty((rows, num_heads, qt, HEAD_DIM), dtype=_FP8, device=q.device)
        qs = torch.empty(
            (rows, num_heads, qt, HEAD_DIM // 32), dtype=torch.uint8, device=q.device
        )
        pairs = rows * num_heads
        if pairs:
            msa_q_mxfp8_kernel[(triton.cdiv(pairs, 64),)](
                q, q8, qs, 1.0, pairs, num_heads, q.stride(0), q.stride(1),
                D=HEAD_DIM, QT=qt, BLOCK=64, num_warps=4,
            )  # fmt: skip
    hg = num_heads // num_kv_heads
    if hg != 16 or num_heads % num_kv_heads:
        raise ValueError("the gluon backend needs 16 query heads per KV head")
    if out is None:
        out = torch.empty((rows, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    if rows == 0:
        return out
    if block_m is None:
        # Decode: one query row per program (no union pre-pass); on real top-k
        # selections (AgentX dumps) neighbouring DSpark rows share only ~10 of
        # 16 pages. Prefill: 2 rows, whose page union is ~1.66x shared.
        block_m = 16 if decode else 32
    if block_m not in (16, 32, 64, 128):
        raise ValueError("block_m must be 16, 32, 64 or 128")
    dev = q.device
    vs = _scale_tensor(v_scale, dev)
    block_q = block_m // hg
    groups = triton.cdiv(rows, block_q)
    if kv_splits is None:
        kv_splits = (
            msa_gluon_splits(groups * num_kv_heads, n_topk, _cus(dev)) if decode else 1
        )
    if kv_splits < 1 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a power of two")
    if block_q == 1 and n_topk % kv_splits:
        raise ValueError("kv_splits must divide TOPK at block_m = 16")
    if block_q > 1:
        # the rows' page union, each page once per row group
        c = block_q * n_topk
        uniq = torch.empty((num_kv_heads, groups, c), dtype=torch.int32, device=dev)
        count = torch.empty((num_kv_heads, groups), dtype=torch.int32, device=dev)
        tokblk = torch.empty((num_kv_heads, groups, c, block_q), dtype=torch.int32, device=dev)
        pos = torch.empty((groups * block_q,), dtype=torch.int32, device=dev)
        msa_union_kernel[(groups, num_kv_heads)](
            topk, block_table,
            seq_lens if decode else query_req, query_req if not decode else seq_lens,
            query_pos if not decode else seq_lens,
            uniq, count, tokblk, pos, rows,
            topk.stride(0), topk.stride(1), block_table.stride(0),
            TOPK=n_topk, TQ=block_q, DECODE_QL=dql if decode else 0, num_warps=4,
        )  # fmt: skip
    else:
        uniq = count = tokblk = pos = topk  # unused at block_m = 16
    if kv_splits > 1:
        n = num_kv_heads * groups * kv_splits
        part_o = torch.empty((n, block_m, HEAD_DIM), dtype=torch.float32, device=dev)
        part_ml = torch.empty((n, block_m, 2), dtype=torch.float32, device=dev)
    else:
        part_o = part_ml = out
    msa_gfx950_gluon_kernel[(groups, num_kv_heads, kv_splits)](
        q8, qs, kv_cache, kv_cache.view(_FP8), topk, block_table,
        seq_lens if decode else query_req, query_req if not decode else seq_lens,
        query_pos if not decode else seq_lens, uniq, count, tokblk, pos, out,
        part_o, part_ml, vs, sm_scale * 1.4426950408889634, rows,
        kv_cache.stride(0), kv_cache.stride(1), topk.stride(0), topk.stride(1),
        block_table.stride(0), out.stride(0), out.stride(1),
        NUM_HEADS=num_heads, TOPK=n_topk, BLOCK_M=block_m, SPLITS=kv_splits,
        DECODE_QL=dql if decode else 0, QT=qt, P_SCALE=P_SCALE,
        NUM_WARPS=num_warps, KV_LDS=kv_lds, num_warps=num_warps,
    )  # fmt: skip
    if kv_splits > 1:
        msa_merge_kernel[(groups, num_kv_heads)](
            part_o, part_ml, out, vs, rows, out.stride(0), out.stride(1),
            BLOCK_M=block_m, HG=hg, D=HEAD_DIM, SPLITS=kv_splits, num_warps=4,
        )  # fmt: skip
    return out


# ----------------------------------------------------------------- k4v4 cache


def msa_k4v4_alloc(num_pages: int, num_kv_heads: int, device) -> torch.Tensor:
    """A zeroed k4v4 cache (K and V MXFP4), uint8 [pages, kv_heads, RECORD4_BYTES]."""
    return torch.zeros(
        (num_pages, num_kv_heads, RECORD4_BYTES), dtype=torch.uint8, device=device
    )


def msa_k4v4_store(k, v, kv_cache, slot_mapping) -> None:
    """Quantize k / v ([tokens, kv_heads, 128]) into their k4v4 slots."""
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[2] != RECORD4_BYTES:
        raise ValueError("kv_cache must be uint8 [pages, kv_heads, RECORD4_BYTES]")
    n = k.shape[0]
    if n == 0:
        return
    block_t = 16
    msa_k4v4_store_kernel[(triton.cdiv(n, block_t), k.shape[1])](
        k, v, slot_mapping, kv_cache, kv_cache.view(torch.int32), n,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        PAGE=PAGE, D=HEAD_DIM, K_BYTES=K_BYTES, KS_BYTES=KS_BYTES,
        V4_BYTES=V4_BYTES, BLOCK_T=block_t, num_warps=4,
    )  # fmt: skip


def _wave_kappa_torch(t: torch.Tensor):
    """(wave, kappa) of page key t in the k4v4 V layout."""
    return (t // 16) % 4, 8 * ((t // 4) % 4) + 4 * (t // 64) + t % 4


def msa_k4v4_dequant(kv_cache: torch.Tensor):
    """Reference unpack: fp32 K and V, each [pages, kv_heads, PAGE, 128]."""
    P, H, _ = kv_cache.shape
    k, _ = msa_k4v8_dequant(
        torch.cat(
            [kv_cache[..., : K_BYTES + KS_BYTES],
             torch.zeros(P, H, V_BYTES, dtype=torch.uint8, device=kv_cache.device)], -1
        ),
        1.0,
    )
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=kv_cache.device)
    vb = kv_cache[..., K_BYTES + KS_BYTES : K_BYTES + KS_BYTES + V4_BYTES]
    vb = vb.reshape(P, H, 4, HEAD_DIM, 16)
    codes = torch.stack([vb & 0xF, vb >> 4], -1).reshape(P, H, 4, HEAD_DIM, 32)  # [w, d, kap]
    w, kap = _wave_kappa_torch(torch.arange(PAGE, device=kv_cache.device))
    vals = lut[codes.long()][:, :, w, :, kap]  # [t, P, H, d]
    sc = kv_cache[..., K_BYTES + KS_BYTES + V4_BYTES :].reshape(P, H, 4, HEAD_DIM // 32, 32)
    sc = torch.exp2(sc.float() - 127.0)[:, :, w, :, kap]  # [t, P, H, g]
    v = vals.reshape(PAGE, P, H, HEAD_DIM // 32, 32) * sc[..., None]
    return k, v.reshape(PAGE, P, H, HEAD_DIM).permute(1, 2, 0, 3).contiguous()


def _k4v4_fwd(
    q, q_scale, kv_cache, topk, block_table, sm_scale, seq_lens, dql,
    query_req, query_pos, out, kv_splits, q_mode,
):  # fmt: skip
    from aiter.ops.triton._gluon_kernels.gfx950.attention.msa import (
        msa_gfx950_gluon_k4v4_kernel,
    )

    rec = RECORD4_BYTES
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3 or kv_cache.shape[2] != rec:
        raise ValueError(f"kv_cache must be k4v4 uint8 [pages, kv_heads, {rec}]")
    if q_mode != "mxfp8":
        raise ValueError("the k4v4 attend takes one MXFP8 Q term (q_mode='mxfp8')")
    rows = q.shape[0]
    num_kv_heads = kv_cache.shape[1]
    decode = seq_lens is not None
    if decode == (query_req is not None):
        raise ValueError("pass seq_lens + decode_query_len (decode) or query_req + query_pos")
    n_topk = topk.shape[2]
    dev = q.device
    if q_scale is not None:
        num_heads = q.numel() // (rows * HEAD_DIM)
        q8 = q.reshape(rows, num_heads * HEAD_DIM)
        qs = q_scale.reshape(rows, num_heads * HEAD_DIM // 32)
    else:
        if q.dtype != torch.bfloat16 or q.ndim != 3 or q.stride(2) != 1:
            raise ValueError("q must be bf16 [rows, H, 128] (or MXFP8 with q_scale)")
        num_heads = q.shape[1]
        q8 = torch.empty((rows, num_heads, HEAD_DIM), dtype=_FP8, device=dev)
        qs = torch.empty((rows, num_heads, HEAD_DIM // 32), dtype=torch.uint8, device=dev)
        pairs = rows * num_heads
        if pairs:
            msa_q_mxfp8_kernel[(triton.cdiv(pairs, 64),)](
                q, q8, qs, 1.0, pairs, num_heads, q.stride(0), q.stride(1),
                D=HEAD_DIM, QT=1, BLOCK=64, num_warps=4,
            )  # fmt: skip
    hg = num_heads // num_kv_heads
    if hg != 16 or num_heads % num_kv_heads:
        raise ValueError("the k4v4 attend needs 16 query heads per KV head")
    if out is None:
        out = torch.empty((rows, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    if rows == 0:
        return out
    if kv_splits is None:
        kv_splits = msa_gluon_splits(rows * num_kv_heads, n_topk, _cus(dev)) if decode else 1
    if kv_splits < 1 or kv_splits & (kv_splits - 1) or n_topk % kv_splits:
        raise ValueError("kv_splits must be a power of two dividing TOPK")
    vs = torch.ones(1, dtype=torch.float32, device=dev)
    if kv_splits > 1:
        n = num_kv_heads * rows * kv_splits
        part_o = torch.empty((n, 16, HEAD_DIM), dtype=torch.float32, device=dev)
        part_ml = torch.empty((n, 16, 2), dtype=torch.float32, device=dev)
    else:
        part_o = part_ml = out
    msa_gfx950_gluon_k4v4_kernel[(rows, num_kv_heads, kv_splits)](
        q8, qs, kv_cache, topk, block_table,
        seq_lens if decode else query_req, query_req if not decode else seq_lens,
        query_pos if not decode else seq_lens, out, part_o, part_ml,
        sm_scale * 1.4426950408889634, kv_cache.stride(0), kv_cache.stride(1),
        topk.stride(0), topk.stride(1), block_table.stride(0), out.stride(0),
        out.stride(1), NUM_HEADS=num_heads, TOPK=n_topk, SPLITS=kv_splits,
        DECODE_QL=dql if decode else 0, num_warps=4,
    )  # fmt: skip
    if kv_splits > 1:
        msa_merge_kernel[(rows, num_kv_heads)](
            part_o, part_ml, out, vs, rows, out.stride(0), out.stride(1),
            BLOCK_M=16, HG=hg, D=HEAD_DIM, SPLITS=kv_splits, num_warps=4,
        )  # fmt: skip
    return out
    if kv_splits is None:
        kv_splits = msa_gluon_splits(rows * num_kv_heads, n_topk, _cus(dev)) if decode else 1
    if kv_splits < 1 or kv_splits & (kv_splits - 1) or n_topk % kv_splits:
        raise ValueError("kv_splits must be a power of two dividing TOPK")
    if kv_splits > 1:
        n = num_kv_heads * rows * kv_splits
        part_o = torch.empty((n, 16, HEAD_DIM), dtype=torch.float32, device=dev)
        part_ml = torch.empty((n, 16, 2), dtype=torch.float32, device=dev)
    else:
        part_o = part_ml = out
    msa_gfx950_gluon_k4v4_kernel[(rows, num_kv_heads, kv_splits)](
        q8, qs, kv_cache, topk, block_table,
        seq_lens if decode else query_req, query_req if not decode else seq_lens,
        query_pos if not decode else seq_lens, out, part_o, part_ml,
        sm_scale * 1.4426950408889634, kv_cache.stride(0), kv_cache.stride(1),
        topk.stride(0), topk.stride(1), block_table.stride(0), out.stride(0),
        out.stride(1), NUM_HEADS=num_heads, TOPK=n_topk, SPLITS=kv_splits,
        DECODE_QL=dql if decode else 0, num_warps=1,
    )  # fmt: skip
    if kv_splits > 1:
        ones = torch.ones(1, dtype=torch.float32, device=dev)
        msa_merge_kernel[(rows, num_kv_heads)](
            part_o, part_ml, out, ones, rows, out.stride(0), out.stride(1),
            BLOCK_M=16, HG=hg, D=HEAD_DIM, SPLITS=kv_splits, num_warps=4,
        )  # fmt: skip
    return out

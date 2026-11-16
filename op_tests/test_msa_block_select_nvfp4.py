# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""nvfp4 MSA index keys: decode and prefill scores.

- Decode scores match a reference on keys dequantized as E4M3(scale * mag).
- nvfp4 prefill scores equal the decode kernel run on every query row as a
  one-token request, for 1, 2 and 4 index heads, every tile count and every
  workgroup mapping, and under max_seq_len bounds far above the contexts.
"""

import argparse
import sys

import torch
from msa_nvfp4_ref import ref_nvfp4_dequant, ref_nvfp4_pages

import aiter
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops import msa_block_select
from aiter.ops.msa_block_select import (
    nvfp4_page_bytes,
    pa_sparse_block_score_decode,
    pa_sparse_block_score_prefill,
)
from aiter.test_common import checkAllclose

BLOCK_SIZE = 128
HEAD_DIM = 128
WAVE = 64
FP8 = torch.float8_e4m3fn
DEV = "cuda"
NEG = -1e4


class _patch:
    """Replace msa_block_select attributes for the duration of a with-block
    (None values are skipped)."""

    def __init__(self, **kv):
        self.kv = {k: v for k, v in kv.items() if v is not None}

    def __enter__(self):
        self.old = {k: getattr(msa_block_select, k) for k in self.kv}
        for k, v in self.kv.items():
            setattr(msa_block_select, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(msa_block_select, k, v)


def _force(wave_tiles=None, order=None):
    """Force the prefill tiles per wave and the order (a function of the
    query group count); None keeps the rule."""
    return _patch(
        _prefill_nvfp4_wave_tiles=None if wave_tiles is None else lambda *a: wave_tiles,
        _prefill_nvfp4_order=None if order is None else lambda *a: order(a[1]),
    )


def _orders(wave_tiles):
    """(name, order) pairs of every prefill order; see nvfp4_prefill_args."""
    xcds = msa_block_select._num_xcds()
    xg = lambda g: min(g & -g, 8) - 1
    out = [
        ("plain", lambda g: (1, 0, 0, 0)),
        ("rot", lambda g: (1, 0, 1, 0)),
        ("xor", lambda g: (1, xg(g), 0, 0)),
        ("xor+rot", lambda g: (1, xg(g), 1, 0)),
        (f"xcds{xcds}", lambda g: (1, 0, 0, xcds)),
        ("xcds3", lambda g: (1, 0, 0, 3)),
    ]
    if wave_tiles == 8:
        out += [
            (f"{name} store2", lambda g, f=f: (2, *f(g)[1:])) for name, f in list(out)
        ]
    return out


def ref_block_scores(q, k, block_table, seq_lens, qlens, num_slots):
    """score[h, n, b] = max over the causal tokens of block b of q[n,h,:] . k[b,t,:]."""
    total_q, heads, _ = q.shape
    ref = torch.full((heads, total_q, num_slots), -float("inf"), device=DEV)
    qf = q.float()
    row = 0
    for r, qlen in enumerate(qlens):
        seq_len = int(seq_lens[r])
        nblk = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[r, :nblk].long()
        prod = torch.einsum("btd,qhd->bqht", k[pages].float(), qf[row : row + qlen])
        kv_len = seq_len - qlen + torch.arange(qlen, device=DEV) + 1
        tok = (
            torch.arange(nblk, device=DEV)[:, None] * BLOCK_SIZE
            + torch.arange(BLOCK_SIZE, device=DEV)[None, :]
        )
        visible = tok[:, None, :] < kv_len[None, :, None]
        prod = prod.masked_fill(~visible[:, :, None, :], -float("inf"))
        ref[:, row : row + qlen, :nblk] = prod.max(dim=-1).values.permute(2, 1, 0)
        row += qlen
    return ref


def _finite(t):
    return torch.nan_to_num(t, neginf=NEG)


def _setup(seq_lens, qlens, num_idx_heads):
    """q, the nvfp4 index cache, the dequantized keys the reference scores
    against, and the launch tensors."""
    num_reqs = len(seq_lens)
    max_blk = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens)
    num_pages = num_reqs * max_blk + 2
    torch.manual_seed(0)
    q = (torch.randn(sum(qlens), num_idx_heads, HEAD_DIM, device=DEV) / 4).to(FP8)
    keys = torch.randn(num_pages * BLOCK_SIZE, HEAD_DIM, device=DEV).to(torch.bfloat16)
    # Outliers, so groups get very different scales.
    keys[::37] *= 64
    cache = ref_nvfp4_pages(keys, BLOCK_SIZE)
    assert cache.shape[1] == nvfp4_page_bytes(BLOCK_SIZE, HEAD_DIM)
    kref = ref_nvfp4_dequant(cache, BLOCK_SIZE, HEAD_DIM)
    block_table = torch.arange(num_reqs * max_blk, device=DEV, dtype=torch.int32).view(
        num_reqs, max_blk
    )
    seq = torch.tensor(seq_lens, device=DEV, dtype=torch.int32)
    slots_w = 1 << (-(-max_blk // WAVE) - 1).bit_length()
    score = torch.full(
        (num_idx_heads, sum(qlens), slots_w * WAVE), -float("inf"), device=DEV
    )
    return q, cache, kref, block_table, seq, score, max_blk


def _check(name, ref, score, max_blk):
    err = checkAllclose(
        _finite(ref[:, :, :max_blk]),
        _finite(score[:, :, :max_blk]),
        msg=name,
        rtol=1e-2,
        atol=1e-2,
    )
    return err == 0


def _decode_rows(q, cache, score, bt, seq, qlens):
    """Prefill reference: each query row as a one-token decode request."""
    qlen = torch.tensor(qlens, device=DEV)
    req = torch.repeat_interleave(torch.arange(len(qlens), device=DEV), qlen)
    start = torch.cumsum(qlen, 0) - qlen
    pos = torch.arange(q.size(0), device=DEV) - start[req]
    kv_len = (seq[req] - qlen[req] + pos + 1).to(torch.int32)
    bt_rows = bt[req].contiguous()
    for h in range(q.size(1)):
        pa_sparse_block_score_decode(
            q[:, h : h + 1].contiguous(),
            cache,
            score[h : h + 1],
            bt_rows,
            kv_len,
            query_len=1,
            max_seq_len=int(seq.max()),
        )


def test_score(num_idx_heads, batch, ctx, query_len):
    fmt = "nvfp4"
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(batch)]
    ok = True

    qlens = [query_len] * batch
    q, cache, kref, bt, seq, score, max_blk = _setup(seq_lens, qlens, num_idx_heads)
    pa_sparse_block_score_decode(
        q, cache, score, bt, seq, query_len=query_len, max_seq_len=max(seq_lens)
    )
    ref = ref_block_scores(q, kref, bt, seq, qlens, score.size(2))
    info = f"H:{num_idx_heads} batch:{batch} ctx:{ctx}"
    ok &= _check(f"[{fmt}] decode {info} qlen:{query_len}", ref, score, max_blk)

    # A mixed batch past one 128-row group, and short queries only.
    for top in (300, 128):
        qlens = [1 + (i * 37) % top for i in range(batch - 1)] + [top]
        q, cache, kref, bt, seq, score, max_blk = _setup(seq_lens, qlens, num_idx_heads)
        cu = torch.tensor([0] + qlens, device=DEV).cumsum(0).to(torch.int32)
        ref = ref_block_scores(q, kref, bt, seq, qlens, score.size(2))
        _decode_rows(q, cache, score, bt, seq, qlens)
        base = score.clone()
        ok &= _check(f"[{fmt}] decode rows {info} qlen<={top}", ref, base, max_blk)
        for wt in (None, 2, 4, 8):
            score.fill_(-float("inf"))
            with _force(wave_tiles=wt):
                pa_sparse_block_score_prefill(
                    q,
                    cache,
                    score,
                    bt,
                    cu,
                    seq,
                    max_query_len=max(qlens),
                    max_seq_len=max(seq_lens),
                )
            if not torch.equal(base, score):
                diff = (_finite(base) - _finite(score)).abs().max().item()
                aiter.logger.info(
                    f"[{fmt}] prefill qlen<={top} wave_tiles={wt} differs from "
                    f"decode rows by up to {diff} -> FAIL"
                )
                ok = False
    aiter.logger.info(
        f"[{fmt}] prefill vs decode rows: {'bit-exact' if ok else 'see above'}"
    )
    return ok


# (prefixes, qlens) for the prefill orders: group and request counts with 1,
# 2, 4 and 8 as largest power-of-two factor.
ORDER_BATCHES = [
    ([19000, 2000], [1024, 1024]),
    ([17000, 9000, 0], [1500, 37, 600]),
    ([16000, 2000, 12000, 500], [300] * 4),
    (
        [9000, 0, 4000, 300, 7000, 100, 11000, 2500],
        [1 + (i * 131) % 900 for i in range(8)],
    ),
    ([(i * 2711) % 12000 for i in range(12)], [1 + (i * 97) % 400 for i in range(12)]),
    # Default plans with the rotated and XOR orders.
    ([50000, 30000, 10000, 0], [1024] * 4),
    ([108000, 40000, 20000, 60000], [2048, 900, 1500, 300]),
    ([58000, 1000, 30000, 200], [1024, 100, 300, 50]),
    # Short queries on contexts far below BOUND_SEQ_LENS.
    ([8188 + i * 61 for i in range(6)], [2, 2, 3, 4, 4, 3]),
    ([(i * 4099) % 33000 for i in range(15)], [1 + i % 4 for i in range(15)]),
    # One request on a long prefix, a cold prefill, contexts up to 131072.
    ([126952], [2048]),
    ([0], [4096]),
    ([131064, 60000, 0], [8, 1, 700]),
    # Contexts up to 262144 and 1048576.
    ([261632], [512]),
    ([1048320], [256]),
    ([261888, 1048000, 3000], [256, 64, 200]),
]

ORDER_HEADS = (1, 2, 4)

# max_seq_len bounds (max model lengths) test_orders also runs at.
BOUND_SEQ_LENS = (131072, 262144, 1048576)

# Orders (store_pages, group_xor, req_rot, xcds); XCD_WIDE gets the device's
# XCD count.
PLAIN = (1, 0, 0, 0)
ROT = (1, 0, 1, 0)
XOR_ROT4 = (1, 3, 1, 0)
XCD_WIDE = (2, 0, 0, None)

# (num_reqs, max_query_len, max_seq_len, total_q) -> (wave_tiles, order).
PLAN_CASES = [
    ((4, 1024, 51024, 4096), (8, ROT)),  # equal lengths, 8 tiles
    ((8, 128, 32896, 1024), (2, PLAIN)),  # equal lengths, 2 tiles
    ((4, 2048, 110048, 4748), (8, XOR_ROT4)),  # ragged, 4 groups
    ((2, 4096, 84096, 7096), (8, ROT)),  # ragged, two requests
    ((8, 4484, 120657, 9197), (8, ROT)),  # ragged, 9 groups (block of 1)
    ((4, 1024, 59024, 1474), (4, XOR_ROT4)),  # ragged, 4 tiles, 4 groups
    ((4, 2504, 111227, 3466), (4, ROT)),  # ragged, 4 tiles, 10 groups
    ((6, 9216, 129216, 11317), (4, ROT)),  # ragged, past XOR_ROT_MAX_QUERY_LEN
    ((4, 956, 956, 3000), (2, ROT)),  # ragged, 2 tiles
    ((3, 564, 115157, 1500), (4, PLAIN)),  # groups x requests not a multiple of 8
]

# (num_reqs, max_query_len, max_seq_len, total_q, heads) ->
# (wave_tiles, num_q_groups, order).
PLAN_HEAD_CASES = [
    ((6, 4, 131072, 18, 2), (2, 1, PLAIN)),  # short queries: one tile per head
    ((6, 4, 131072, 18, 4), (4, 1, PLAIN)),
    ((1, 2048, 2048, 2048, 2), (2, 32, PLAIN)),  # one request, little work
    ((1, 2048, 94208, 2048, 2), (8, 8, XCD_WIDE)),  # 4096 rows on a long prefix
    ((1, 4096, 96256, 4096, 2), (8, 16, XCD_WIDE)),
    ((1, 2048, 94208, 2048, 4), (8, 16, XCD_WIDE)),
    ((1, 8192, 100352, 8192, 1), (8, 16, XCD_WIDE)),
    ((4, 1024, 51024, 4096, 2), (8, 4, ROT)),  # equal lengths
]


def test_plan_rule():
    """_score_prefill_nvfp4_plan on representative batches."""
    xcds = msa_block_select._num_xcds()

    def device(order):
        return order[:3] + (xcds,) if order[3] is None else order

    cases = [(*key, 1) for key, _ in PLAN_CASES] + [key for key, _ in PLAN_HEAD_CASES]
    wants = [(wt, None, o) for _, (wt, o) in PLAN_CASES] + [
        want for _, want in PLAN_HEAD_CASES
    ]
    ok = True
    for (nreq, max_q, max_seq, total_q, heads), (want_wt, want_g, want_o) in zip(
        cases, wants
    ):
        max_blk = -(-max_seq // BLOCK_SIZE)
        wt, groups, _, _, order = msa_block_select._score_prefill_nvfp4_plan(
            nreq, max_q, max_blk, total_q, heads
        )
        got = (wt, groups if want_g is not None else None, order)
        want = (want_wt, want_g, device(want_o))
        if got != want:
            aiter.logger.info(
                f"plan B:{nreq} max_q:{max_q} max_seq:{max_seq} total_q:{total_q}"
                f" H:{heads} -> (wave_tiles, groups, order) {got}, expected {want}"
                " -> FAIL"
            )
            ok = False
    aiter.logger.info(f"nvfp4 prefill plan rule: {'PASS' if ok else 'FAIL'}")
    return ok


def test_grid_chunks():
    """Model of the per-request chunk rule of the prefill kernel: it matches
    the host split, the grid covers every request, idle chunks are only past
    the request's end, and the grid stays within its cap."""
    m = msa_block_select
    target = m._prefill_nvfp4_target_workgroups()
    batch_wg = get_cu_num() * m._NVFP4_PREFILL_BATCH_WG_PER_CU
    ok = True
    for groups in (1, 2, 3, 6, 15, 62, 64, 255, 703, 704, 705, 2000):
        req = m._prefill_nvfp4_req_chunks(groups)
        small = 2 * (req - 1)
        for max_blk in (1, 7, 64, 257, 1024, 2048, 8192):
            chunks, chunk = m._score_split_prefill_nvfp4(max_blk, groups)
            grid = m._prefill_nvfp4_grid_chunks(max_blk, groups, chunks)
            need, same, idle = 0, True, True
            for nb in range(1, max_blk + 1):
                cb = 1
                while (cb < chunk and nb > 2 * cb * (req - 1)) or cb * grid < nb:
                    if cb < chunk and nb > 2 * cb * (req - 1):
                        same &= groups * -(-nb // (2 * cb)) >= target
                    cb *= 2
                if cb < chunk:
                    same &= groups * -(-nb // (2 * cb)) < target
                need = max(need, -(-nb // cb))
                idle &= max(-(-nb // chunk), small) * cb >= nb
            per_req = max(m._NVFP4_PREFILL_BATCH_CHUNKS, small)
            big = groups * grid > max(batch_wg, groups * per_req)
            if not (same and idle) or big or need > grid:
                aiter.logger.info(
                    f"grid chunks groups:{groups} max_blk:{max_blk}: grid {grid}, "
                    f"need {need}, same {same}, idle {idle} -> FAIL"
                )
                ok = False
    aiter.logger.info(f"nvfp4 prefill grid chunks: {'PASS' if ok else 'FAIL'}")
    return ok


def test_orders(prefixes, qlens, num_heads=1):
    """Default plan, every order and tile count, and max_seq_len bounds
    against the decode rows reference."""
    fmt = "nvfp4"
    seq_lens = [p + t for p, t in zip(prefixes, qlens)]
    q, cache, kref, bt, seq, score, max_blk = _setup(seq_lens, qlens, num_heads)
    cu = torch.tensor([0] + qlens, device=DEV).cumsum(0).to(torch.int32)
    max_seq = max(seq_lens)
    runs = [(None, "rule", None, max_seq)]
    runs += [
        (wt, name, order, max_seq)
        for wt in (2, 4, 8)
        if wt >= num_heads
        for name, order in _orders(wt)
    ]
    runs += [(None, "rule", None, b) for b in BOUND_SEQ_LENS if b >= max_seq]
    info = f"H:{num_heads} B:{len(qlens)} max_q:{max(qlens)} max_seq:{max(seq_lens)}"

    _decode_rows(q, cache, score, bt, seq, qlens)
    base = score.clone()
    ref = ref_block_scores(q, kref, bt, seq, qlens, score.size(2))
    ok = _check(f"[{fmt}] decode rows {info}", ref, base, max_blk)
    for wt, name, order, bound in runs:
        score.fill_(-float("inf"))
        with _force(wave_tiles=wt, order=order):
            pa_sparse_block_score_prefill(
                q,
                cache,
                score,
                bt,
                cu,
                seq,
                max_query_len=max(qlens),
                max_seq_len=bound,
            )
        if not torch.equal(base, score):
            aiter.logger.info(
                f"[{fmt}] prefill {info} wave_tiles={wt} order={name} "
                f"max_seq_len={bound} differs from decode rows -> FAIL"
            )
            ok = False
    aiter.logger.info(
        f"[{fmt}] prefill {info}: {len(runs)} runs vs decode rows: "
        f"{'bit-exact' if ok else 'see above'}"
    )
    return ok


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("-H", "--num_idx_heads", type=int, nargs="*", default=[1, 2, 4])
parser.add_argument("-b", "--batch", type=int, default=4)
parser.add_argument("-c", "--ctx", type=int, default=4096)
parser.add_argument("-q", "--query_len", type=int, nargs="*", default=[4, 8])
args = parser.parse_args()

current_gfx = aiter.get_gfx()
if current_gfx != "gfx950":
    print(f"Skipping {__file__}: requires gfx950, got {current_gfx}")
    sys.exit(0)

results = [test_plan_rule(), test_grid_chunks()]
for h in args.num_idx_heads:
    for ql in args.query_len:
        results.append(test_score(h, args.batch, args.ctx, ql))
for h in ORDER_HEADS:
    for prefixes, qlens in ORDER_BATCHES:
        results.append(test_orders(prefixes, qlens, h))
print("ALL PASS" if all(results) else "FAILURES")
sys.exit(0 if all(results) else 1)

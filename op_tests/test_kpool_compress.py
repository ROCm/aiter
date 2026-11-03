# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy (and timing) of the GLM-5.3-Flash indexer kpool cache writers:
``kpool_compress_insert``, ``kpool_decode_update`` and ``kpool_mixed_write``.

Every case runs a pure-torch reference of the vLLM semantics (prefill insert,
``kpool_seed_tail_cache``, the ordered decode walk) and compares the whole
kv_cache (dequantized fp8 keys and fp32 scales) and the whole tail cache.
The reference rounds at the same two bf16 points as the kernel; the only
sources of difference are 1-ulp fp32 choices (exp, division, log2), so the
dequantized keys agree to one fp8 ulp and the bytes agree exactly for all but
a handful of tokens. With ``--vllm-ref <path to vllm kpool_compress.py>`` the
Triton production kernels are run too and compared byte for byte.

    python3 test_kpool_compress.py                 # all three writers
    python3 test_kpool_compress.py --only insert   # insert | decode | mixed
    python3 test_kpool_compress.py --quick
"""

import argparse
import importlib.util
import math
import sys

import pandas as pd
import torch

import aiter
from aiter import get_gfx
from aiter.ops.kpool_compress import kpool_pick_lanes
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

HEAD_DIM = 128
FP8_MAX = 448.0
HAD_SCALE = 0.08838834764831845  # 1/sqrt(128), the fp32 constant of the kernels
FP8 = torch.float8_e4m3fn
TILE = 16

# --------------------------------------------------------------------------
# torch reference of the pool math and the cache layout
# --------------------------------------------------------------------------


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard, stages of increasing stride (same order as the kernels)."""
    n = x.shape[-1]
    h = 1
    while h < n:
        v = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape)
        h *= 2
    return x


def compress_pools(k_rows, g_rows, ape, round_scale=True):
    """k_rows, g_rows [n, pool, 128] bf16; ape [pool, 128] fp32 ->
    fp8 bytes [n, 128] uint8 and fp32 scale [n]."""
    u = g_rows.float() + ape.float()[None]
    m = u.amax(dim=1, keepdim=True)
    e = torch.exp(u - m)
    z = (e * k_rows.float()).sum(1) / e.sum(1)
    x = z.to(torch.bfloat16).float()
    y = (hadamard128(x) * HAD_SCALE).to(torch.bfloat16).float()
    a = y.abs().amax(1).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(a / FP8_MAX)))
    else:
        scale = a / FP8_MAX
    q = (y / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(torch.uint8), scale


def _k_offsets(tok, page):
    """Byte offsets of the 128 keys of token(s) ``tok`` inside a page."""
    d = torch.arange(HEAD_DIM)
    if page > 1:
        return (
            (tok // TILE)[..., None] * (TILE * HEAD_DIM)
            + (d // TILE)[None] * (TILE * TILE)
            + (tok % TILE)[..., None] * TILE
            + (d % TILE)[None]
        )
    return d[None].expand(*tok.shape, HEAD_DIM)


def write_cache(kv, loc, q_u8, scale):
    """Store fp8 rows and scales at paged slots ``loc`` (int64, all >= 0)."""
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    pg, tok = loc // page, loc % page
    flat[pg[:, None], _k_offsets(tok, page)] = q_u8
    soff = page * HEAD_DIM + tok[:, None] * 4 + torch.arange(4)[None]
    flat[pg[:, None], soff] = scale.contiguous().view(torch.uint8).view(-1, 4)


def read_cache(kv):
    """kv_cache -> dequantized keys [nb, page, 128] fp32 and scales [nb, page]."""
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    kbytes = flat[:, : page * HEAD_DIM]
    k = kbytes[:, _k_offsets(torch.arange(page), page)]  # [nb, page, 128]
    scale = flat[:, page * HEAD_DIM :].contiguous().view(torch.float32).view(nb, page)
    return k.view(FP8).float() * scale[..., None], scale


def ref_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = k.shape[0]
    t = torch.arange(n)
    closing = (slot >= 0) & (t >= pool - 1)
    tc = t[closing]
    if tc.numel() == 0:
        return
    idx = tc[:, None] - (pool - 1) + torch.arange(pool)[None]
    q, s = compress_pools(k[idx], gate[idx], ape, round_scale)
    write_cache(kv, slot[closing].long(), q, s)


def ref_seed(tail, k, gate, tail_slot, pool):
    """kpool_seed_tail_cache: a token is among the last ``pool`` tokens of its
    request when the token ``pool`` ahead has no tail slot or another block."""
    n = k.shape[0]
    ring = tail.shape[2]
    ts = tail_slot.long()
    ahead = torch.full_like(ts, -1)
    if n > pool:
        ahead[: n - pool] = ts[pool:]
    write = (ts >= 0) & ((ahead < 0) | (ahead // ring != ts // ring))
    idx = write.nonzero().flatten()
    tail[ts[idx] // ring, 0, ts[idx] % ring] = k[idx]
    tail[ts[idx] // ring, 1, ts[idx] % ring] = gate[idx]


def ref_decode(
    kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, round_scale=True
):
    """The ordered walk: per request, token by token, read ring then stash."""
    B, next_n = key.shape[:2]
    ring = tail.shape[2]
    ts_c, sm_c, pos_c = tail_slot.cpu(), slot_map.cpu(), pos.cpu()
    for b in range(B):
        for i in range(next_n):
            p, ts, loc = int(pos_c[b, i]), int(ts_c[b, i]), int(sm_c[b, i])
            if p >= 0 and loc >= 0 and p % pool == pool - 1:
                blk, slot = max(ts, 0) // ring, p % pool
                k_rows = torch.empty(pool, HEAD_DIM, dtype=torch.bfloat16)
                g_rows = torch.empty_like(k_rows)
                for s in range(pool):
                    if s == slot:
                        k_rows[s], g_rows[s] = key[b, i], score[b, i]
                    else:
                        phys = (p - slot + s) % ring
                        k_rows[s], g_rows[s] = tail[blk, 0, phys], tail[blk, 1, phys]
                q, sc = compress_pools(k_rows[None], g_rows[None], ape, round_scale)
                write_cache(kv, torch.tensor([loc]), q, sc)
            if p >= 0 and ts >= 0:
                tail[ts // ring, 0, p % ring] = key[b, i]
                tail[ts // ring, 1, p % ring] = score[b, i]


# --------------------------------------------------------------------------
# optional byte-exact reference: the vLLM Triton kernels
# --------------------------------------------------------------------------

_VLLM = None


def load_vllm_ref(path):
    global _VLLM
    spec = importlib.util.spec_from_file_location("kpool_compress_vllm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kpool_compress_vllm"] = mod
    spec.loader.exec_module(mod)
    _VLLM = mod


def vllm_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = slot.shape[0]
    if n < pool:
        return
    t = torch.arange(n)
    write_mask = (slot >= 0) & (t >= pool - 1)
    idx = (t - (pool - 1)).clamp_min(0)[:, None] + torch.arange(pool)[None]
    _VLLM.kpool_compress_and_write_cache(
        kv,
        k[idx],
        gate[idx],
        ape,
        slot.to(torch.int64),
        pool_size=pool,
        head_dim=HEAD_DIM,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def vllm_decode(
    kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, round_scale=True
):
    # one request at a time: the batched launch is not deterministic at B >= 1024
    for b in range(key.shape[0]):
        _VLLM.kpool_decode_update_and_maybe_write_cache_batched(
            kv,
            tail,
            tail_slot[b : b + 1],
            key[b : b + 1],
            score[b : b + 1],
            ape,
            slot_map[b : b + 1],
            pos[b : b + 1],
            pool,
            HEAD_DIM,
            round_scale=round_scale,
        )


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def make_prefill(
    lens, pool, page, slot_dtype=torch.int64, seed=0, ring=None, nblock_offset=0
):
    """Varlen prefill batch, pools aligned to each request start, random paged
    slots. With ``ring`` also returns the tail slot mapping for seeding."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots, tslots = [], []
    n_pools = sum(length // pool for length in lens)
    num_blocks = max(1, math.ceil(n_pools / page)) + 2
    perm = torch.randperm(num_blocks * page, device="cuda", generator=g)[
        :n_pools
    ].tolist()
    for r, length in enumerate(lens):
        for t in range(length):
            slots.append(perm.pop() if t % pool == pool - 1 else -1)
            tslots.append((nblock_offset + r) * (ring or pool) + t % (ring or pool))
    n = len(slots)
    k = torch.randn(n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    gate = (torch.randn(n, HEAD_DIM, generator=g, device="cuda") * 3).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device="cuda")
    slot = torch.tensor(slots, dtype=slot_dtype)
    tslot = torch.tensor(
        tslots, dtype=slot_dtype
    )  # the seed slots share the slot_mapping dtype
    kv = torch.zeros(num_blocks, page, HEAD_DIM + 4, dtype=torch.uint8)
    return k, gate, ape, slot, tslot, kv


def make_decode(
    B,
    next_n,
    pool,
    page=64,
    seed=0,
    ring=None,
    nblocks=None,
    idx_dtype=torch.int32,
    tail_pad=1,
):
    """B requests x next_n consecutive tokens from random context lengths.
    ``idx_dtype`` (int32 / int64) is the dtype of the three metadata tensors
    (vLLM keeps its slot mappings and positions in int64). ``tail_pad`` > 1
    pads every tail page to ``tail_pad`` rings (vLLM pads the tail page to the
    KV-cache page, so stride(0) >> 2 * ring * 128 and the tail can exceed 2^31
    elements): the tail is then a strided view and the requests are spread
    over all blocks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ring = ring or pool
    nblocks = nblocks or B + 2
    tail_base = torch.randn(
        nblocks, 2, ring * tail_pad, HEAD_DIM, generator=g, device="cuda"
    ).bfloat16()
    tail = tail_base[:, :, :ring] if tail_pad > 1 else tail_base
    ctx = torch.randint(pool, 100000, (B,), generator=g, device="cuda")
    pos = (ctx[:, None] + torch.arange(next_n)[None]).to(torch.int32)
    spread = max(1, (nblocks - 1) // B) if tail_pad > 1 else 1
    blocks = (torch.arange(B, dtype=torch.int32) * spread)[:, None]
    tail_slot = (blocks * ring + pos % ring).to(torch.int32)
    n_pages = math.ceil(B * next_n / page) + 2
    locs = torch.randperm(n_pages * page, generator=g, device="cuda")[: B * next_n]
    locs = locs.view(B, next_n).to(torch.int32)
    completes = pos % pool == pool - 1
    slot_map = torch.where(completes, locs, torch.full_like(locs, -1))
    key = torch.randn(B, next_n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    score = torch.randn(B, next_n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device="cuda")
    kv = torch.zeros(n_pages, page, HEAD_DIM + 4, dtype=torch.uint8)
    return {
        "kv": kv,
        "tail": tail,
        "tail_base": tail_base,
        "tail_slot": tail_slot.to(idx_dtype),
        "key": key,
        "score": score,
        "ape": ape,
        "slot_map": slot_map.to(idx_dtype),
        "pos": pos.to(idx_dtype),
        "pool": pool,
    }


def tail_clone(d):
    """Copy of the tail that keeps a padded tail's strides (clone() would compact the view)."""
    base = d["tail_base"].clone()
    return (
        base[:, :, : d["tail"].shape[2]]
        if d["tail"].shape[2] != base.shape[2]
        else base
    )


def compare_cache(kv_ref, kv_got, msg):
    y_ref, s_ref = read_cache(kv_ref)
    y_got, s_got = read_cache(kv_got)
    err = checkAllclose(
        y_ref, y_got, rtol=0.13, atol=2e-3, msg=f"{msg} keys (dequantized)"
    )
    err_s = checkAllclose(s_ref, s_got, rtol=1e-6, atol=0, msg=f"{msg} scales")
    diff_bytes = int((kv_ref != kv_got).sum())
    return err, err_s, diff_bytes


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@benchmark()
def test_insert(lens, pool, page, slot_dtype=torch.int64, lanes=0, round_scale=True):
    k, gate, ape, slot, _, kv = make_prefill(lens, pool, page, slot_dtype)
    n = k.shape[0]
    ref = kv.clone()
    ref_insert(ref, k, gate, ape, slot, pool, round_scale)
    got = kv.clone()
    _, us = run_perftest(
        aiter.kpool_compress_insert,
        got,
        k,
        gate,
        ape,
        slot,
        pool,
        round_scale,
        lanes,
        num_iters=20,
        num_warmup=2,
    )
    err, err_s, diff_bytes = compare_cache(ref, got, "insert")
    ret = {
        "tokens": n,
        "lanes": lanes or kpool_pick_lanes(n, pool),
        "err": err,
        "scale err": err_s,
        "diff bytes": diff_bytes,
        "us": us,
        "TB/s (K+gate)": n * 2 * HEAD_DIM * 2 / us / 1e6,
    }
    if _VLLM is not None:
        exact = kv.clone()
        vllm_insert(exact, k, gate, ape, slot, pool, round_scale)
        ret["vllm diff bytes"] = int((exact != got).sum())
    return ret


@benchmark()
def test_decode(
    B,
    next_n,
    pool,
    page=64,
    round_scale=True,
    ring=None,
    idx_dtype=torch.int32,
    tail_pad=1,
    nblocks=None,
):
    d = make_decode(
        B,
        next_n,
        pool,
        page,
        seed=B * 100 + next_n,
        ring=ring,
        idx_dtype=idx_dtype,
        tail_pad=tail_pad,
        nblocks=nblocks,
    )
    kv_ref, tail_ref = d["kv"].clone(), tail_clone(d)
    ref_decode(
        kv_ref,
        tail_ref,
        d["tail_slot"],
        d["key"],
        d["score"],
        d["ape"],
        d["slot_map"],
        d["pos"],
        pool,
        round_scale,
    )
    # accuracy on one call: with next_n > 1 the update is not idempotent (a
    # second run reads ring slots the first run's stashes overwrote)
    kv_got, tail_got = d["kv"].clone(), tail_clone(d)
    aiter.kpool_decode_update(
        kv_got,
        tail_got,
        d["tail_slot"],
        d["key"],
        d["score"],
        d["ape"],
        d["slot_map"],
        d["pos"],
        pool,
        round_scale,
    )
    torch.cuda.synchronize()
    err, err_s, diff_bytes = compare_cache(kv_ref, kv_got, "decode")
    tail_ok = torch.equal(tail_ref, tail_got)
    # timing on scratch copies
    _, us = run_perftest(
        aiter.kpool_decode_update,
        d["kv"].clone(),
        tail_clone(d),
        d["tail_slot"],
        d["key"],
        d["score"],
        d["ape"],
        d["slot_map"],
        d["pos"],
        pool,
        round_scale,
        num_iters=20,
        num_warmup=2,
    )
    ret = {
        "ring": d["tail"].shape[2],
        "idx": str(idx_dtype).replace("torch.", ""),
        "tail stride0": d["tail"].stride(0),
        "err": err,
        "scale err": err_s,
        "diff bytes": diff_bytes,
        "tail exact": tail_ok,
        "us": us,
    }
    assert tail_ok, "tail ring mismatch"
    if _VLLM is not None:
        if d["tail"].shape[2] == pool:
            kv_e, tail_e = d["kv"].clone(), tail_clone(d)
            vllm_decode(
                kv_e,
                tail_e,
                d["tail_slot"],
                d["key"],
                d["score"],
                d["ape"],
                d["slot_map"],
                d["pos"],
                pool,
                round_scale,
            )
            ret["vllm diff bytes"] = int((kv_e != kv_got).sum()) + int(
                (tail_e != tail_got).sum()
            )
        else:
            # the vendored vLLM walk asserts ring == pool; production rings are
            # checked byte for byte against Anh Cao's kpool_decode_update_parallel
            # (tools / cmp_cao.py) instead
            ret["vllm diff bytes"] = -1
    return ret


@benchmark()
def test_mixed(
    lens, B, next_n, pool, page=64, round_scale=True, idx_dtype=torch.int64, tail_pad=1
):
    ring = pool
    # one index dtype per launch: prefill slot_mapping, seed slots and decode metadata
    d = make_decode(
        B,
        next_n,
        pool,
        page,
        seed=7,
        ring=ring,
        nblocks=B + len(lens) + 2,
        idx_dtype=idx_dtype,
        tail_pad=tail_pad,
    )
    k, gate, ape, slot, tslot, kv_p = make_prefill(
        lens, pool, page, idx_dtype, seed=11, ring=ring, nblock_offset=B
    )
    # prefill cache slots live in their own pages, after the decode pages
    slot = torch.where(slot >= 0, slot + d["kv"].shape[0] * page, slot)
    kv = torch.cat([d["kv"], kv_p], 0)
    ape = d["ape"]
    kv_ref, tail_ref = kv.clone(), tail_clone(d)
    ref_insert(kv_ref, k, gate, ape, slot, pool, round_scale)
    ref_seed(tail_ref, k, gate, tslot, pool)
    ref_decode(
        kv_ref,
        tail_ref,
        d["tail_slot"],
        d["key"],
        d["score"],
        ape,
        d["slot_map"],
        d["pos"],
        pool,
        round_scale,
    )
    kv_got, tail_got = kv.clone(), tail_clone(d)
    aiter.kpool_mixed_write(
        kv_got,
        ape,
        pool,
        round_scale,
        0,
        k,
        gate,
        slot,
        tslot,
        tail_got,
        d["tail_slot"],
        d["key"],
        d["score"],
        d["slot_map"],
        d["pos"],
    )
    torch.cuda.synchronize()
    err, err_s, diff_bytes = compare_cache(kv_ref, kv_got, "mixed")
    _, us = run_perftest(
        aiter.kpool_mixed_write,
        kv.clone(),
        ape,
        pool,
        round_scale,
        0,
        k,
        gate,
        slot,
        tslot,
        tail_clone(d),
        d["tail_slot"],
        d["key"],
        d["score"],
        d["slot_map"],
        d["pos"],
        num_iters=20,
        num_warmup=2,
    )
    tail_ok = torch.equal(tail_ref, tail_got)
    assert tail_ok, "tail ring mismatch"
    return {
        "prefill tokens": k.shape[0],
        "err": err,
        "scale err": err_s,
        "diff bytes": diff_bytes,
        "tail exact": tail_ok,
        "us": us,
    }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test the GLM-5.3-Flash kpool cache writers.",
)
parser.add_argument("--only", choices=["insert", "decode", "mixed"], default=None)
parser.add_argument("--quick", action="store_true", help="fewer and smaller cases")
parser.add_argument(
    "--pool", type=int, nargs="*", default=[4, 2, 3, 8, 16], help="pool sizes"
)
parser.add_argument(
    "--page", type=int, nargs="*", default=[64, 1], help="cache page sizes"
)
parser.add_argument(
    "--vllm-ref",
    type=str,
    default=None,
    help="path to vllm's kpool_compress.py: also compare byte for byte",
)
args = parser.parse_args()

if get_gfx() != "gfx950":
    aiter.logger.info(
        "kpool writers target gfx950 (fp8 e4m3fn, permlane swaps); skipping"
    )
    sys.exit(0)
if args.vllm_ref:
    load_vllm_ref(args.vllm_ref)

tables = {}
if args.only in (None, "insert"):
    rows = []
    for pool in args.pool:
        for page in args.page:
            rows.append(test_insert([1024], pool, page))
    for lanes in (16, 32, 64):
        rows.append(test_insert([2048], 4, 64, lanes=lanes))
    rows.append(test_insert([1, 7, 130, 1000, 33, 4, 5], 4, 32, slot_dtype=torch.int32))
    rows.append(test_insert([4099, 17, 2048, 3], 3, 32))
    rows.append(test_insert([300, 77, 2048], 4, 64, round_scale=False))
    if not args.quick:
        rows.append(test_insert([8192] * 8, 4, 64))
        rows.append(test_insert([131072] * 2, 4, 64))
    tables["insert"] = pd.DataFrame(rows)
if args.only in (None, "decode"):
    rows = []
    for B, next_n, pool in (
        (4, 1, 4),
        (3, 4, 4),
        (8, 2, 4),
        (5, 8, 4),
        (2, 4, 16),
        (16, 3, 8),
    ):
        rows.append(test_decode(B, next_n, pool))
    # production rings (vLLM: pool * pow2(ceil((pool + next_n - 1) / pool))): the
    # regular decode path; ring == pool above exercises the generic one
    for B, next_n, ring in (
        (8, 2, 8),
        (6, 3, 8),
        (5, 4, 8),
        (4, 5, 8),
        (4, 6, 16),
        (4, 7, 16),
        (5, 8, 16),
        (3, 16, 32),
    ):
        rows.append(test_decode(B, next_n, 4, ring=ring))
    # int64 metadata (vLLM): same kernels with an 8 B element stride
    for B, next_n, ring in ((4, 1, 4), (5, 4, 8), (3, 2, 4), (5, 8, 16)):
        rows.append(test_decode(B, next_n, 4, ring=ring, idx_dtype=torch.int64))
    if not args.quick:
        rows.append(test_decode(64, 4, 4))
        rows.append(test_decode(32, 16, 4))
        rows.append(test_decode(64, 1, 4, idx_dtype=torch.int64))
        # vLLM layout: tail pages padded to the KV page (strided view), offsets past 2^31 elements
        rows.append(test_decode(64, 1, 4, idx_dtype=torch.int64, tail_pad=16))
        rows.append(test_decode(5, 4, 4, ring=8, idx_dtype=torch.int64, tail_pad=16))
        rows.append(
            test_decode(512, 1, 4, idx_dtype=torch.int64, tail_pad=1024, nblocks=4096)
        )  # 8.6 GB, stride0 2^20
        rows.append(
            test_decode(
                256, 2, 4, ring=8, idx_dtype=torch.int64, tail_pad=512, nblocks=4096
            )
        )
    tables["decode"] = pd.DataFrame(rows)
if args.only in (None, "mixed"):
    rows = [
        test_mixed([64, 9], 1, 1, 4),
        test_mixed([4096, 17, 300], 8, 4, 4),
        test_mixed([64, 9], 1, 1, 4, idx_dtype=torch.int32),
        test_mixed([300, 17], 8, 4, 4, idx_dtype=torch.int32),
    ]
    if not args.quick:
        rows.append(test_mixed([2048] * 4, 32, 2, 4))
        # vLLM layout: padded tail pages (strided view) for the seed + decode roles
        rows.append(test_mixed([4096, 17, 300], 8, 1, 4, tail_pad=16))
        rows.append(test_mixed([300, 77], 256, 1, 4, tail_pad=512))
    tables["mixed"] = pd.DataFrame(rows)
for name, df in tables.items():
    aiter.logger.info(
        "kpool %s summary (markdown):\n%s", name, df.to_markdown(index=False)
    )

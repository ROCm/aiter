# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""IOD-local placement of an A8K4V8 paged KV cache on gfx1250 (SPX + NPS1).

A 16 KiB granule is homed on IOD label a14 ^ a17 ^ a20 ^ phase(2 MiB page); the phase is probed once per region
(aiter.ops.flydsl.kernels.iod_loc_probe). Labels are the probe's; a workgroup on AID a has label a ^ label_xor.

One layer's cache is two regions, the K slice and the V slice of the stock pool, each re-laid in 2 MiB pages:

  block code c (the scheduler's block id): domain d = c & 1, index n = c >> 1 inside the domain.
  K region: page pk = n // 28, slot s = n % 28, x = d ^ phase_k[pk]
    K data of (c, head h) = granule gran(pk, x, 2 s + h)            (the stock page's bytes 0 .. 16 KiB)
    K scales of (c, h)    = 2 KiB row r = 2 s + h of granule gran(pk, x, 56 + r // 8), at (r % 8) * 2 KiB
                                                                     (the stock page's bytes 16 .. 18 KiB)
  V region: page pv = n // 16, slot s = n % 16, x = d ^ phase_v[pv]
    V half q of (c, h)    = granule gran(pv, x, 4 s + 2 h + q)      (the stock page's bytes q * 16 KiB ..)
  gran(page, x, u) = 128 page + 2 u + (x ^ (u >> 2 & 1) ^ (u >> 5 & 1)): the u-th of the page's granules homed on d.

Every granule of a block is homed on its domain. Kernels read x per code from a [codes] uint8 table
(bit 0: K region, bit 1: V region), one table per layer, so one block table serves all layers.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger("aiter")

PAGE = 2 << 20
GRAN = 16384
K_PER_PAGE = 28  # blocks per domain in a K page: 56 data granules + 7 scale granules
V_PER_PAGE = 16  # blocks per domain in a V page: 64 granules
HEADS = 2
PROBE_TRIES = 4  # probe_phase runs, each after a cache eviction
EVICT_BYTES = (
    1 << 30
)  # a read sweep this large evicts the lines an earlier probe left in L2 / MALL


def gran(page, x, u):
    return (page << 7) + (u << 1) + (x ^ ((u >> 2) & 1) ^ ((u >> 5) & 1))


def _aligned(region: torch.Tensor):
    """The 2 MiB-aligned whole pages of a byte region."""
    b = region.view(torch.uint8).reshape(-1)
    off = (-b.data_ptr()) % PAGE
    npg = (b.numel() - off) // PAGE
    return b[off : off + npg * PAGE], npg


def _evict(device):
    """Read a large buffer through the caches. probe_phase times one cold load per granule and launch, on lines an
    earlier probe of the same memory (a buffer the allocator hands out again) may have left in L2 / MALL: those
    read warm on either IOD, the latency pattern is lost and the probe gives up. Writes do not evict them.
    """
    junk = torch.empty(EVICT_BYTES // 4, dtype=torch.int32, device=device)
    junk.sum()
    torch.cuda.synchronize(device)


def probe(buf: torch.Tensor, tries: int = PROBE_TRIES):
    """(page phases, label_xor) of a 2 MiB-aligned buffer (iod_loc_probe), each try after a cache eviction.
    None if every try failed.
    """
    from aiter.ops.flydsl.kernels.iod_loc_probe import probe_phase

    last = None
    for _ in range(tries):
        _evict(buf.device)
        ph, _weak, agree, rot = probe_phase(buf, 4)
        if ph is not None and probe_phase.label_xor is not None:
            return np.asarray(ph, dtype=np.int64), int(probe_phase.label_xor)
        last = (agree, rot)
    logger.warning(
        "[kv_iod] IOD probe failed %d times (last: agreement %s, rotations %s)",
        tries,
        *last
    )
    return None


class KvIodLayer:
    """One layer's placed K and V regions: aligned bases, page phases, the x table.

    A failed probe (``probe``) leaves the layout usable: phase 0 for every page and ``placed = False``. Every
    kernel still reads exact data (a page's slots are a bijection whatever the phase); only the homes are
    unknown, so callers run the stock workgroup order (kv_loc mode 1)."""

    def __init__(self, k_region: torch.Tensor, v_region: torch.Tensor):
        self.k, self.k_pages = _aligned(k_region)
        self.v, self.v_pages = _aligned(v_region)
        rk, rv = probe(self.k), probe(self.v)
        self.placed = rk is not None and rv is not None and rk[1] == rv[1]
        if self.placed:
            (self.phase_k, self.label_xor), (self.phase_v, _) = rk, rv
        else:
            self.phase_k = np.zeros(self.k_pages, dtype=np.int64)
            self.phase_v = np.zeros(self.v_pages, dtype=np.int64)
            self.label_xor = 0
        self.per_domain = min(self.k_pages * K_PER_PAGE, self.v_pages * V_PER_PAGE)

    @property
    def codes(self) -> int:
        return 2 * self.per_domain

    def x_table(self, codes: int, device) -> torch.Tensor:
        c = np.arange(codes)
        d, n = c & 1, c >> 1
        xk = d ^ self.phase_k[np.minimum(n // K_PER_PAGE, self.k_pages - 1)]
        xv = d ^ self.phase_v[np.minimum(n // V_PER_PAGE, self.v_pages - 1)]
        return torch.from_numpy((xk | (xv << 1)).astype(np.uint8)).to(device)


def placed_offsets(codes: np.ndarray, x: np.ndarray, h: int):
    """Byte offsets of (K data, K scales, V half 0, V half 1) of blocks `codes`, head h, given their x bytes."""
    n = codes >> 1  # the domain (codes & 1) is in x
    xk, xv = (x & 1).astype(np.int64), ((x >> 1) & 1).astype(np.int64)
    pk, sk = n // K_PER_PAGE, n % K_PER_PAGE
    pv, sv = n // V_PER_PAGE, n % V_PER_PAGE
    kd = gran(pk, xk, 2 * sk + h) * GRAN
    r = 2 * sk + h
    ks = gran(pk, xk, 56 + r // 8) * GRAN + (r % 8) * 2048
    v0 = gran(pv, xv, 4 * sv + 2 * h) * GRAN
    v1 = gran(pv, xv, 4 * sv + 2 * h + 1) * GRAN
    return kd, ks, v0, v1


def stock_to_placed(k_stock, v_stock, layer: KvIodLayer, codes, x_table):
    """Copy stock pages ([blocks, 2, 128, 144] uint8 K, [blocks, 2, ...] fp8 V; block b -> code codes[b]) into the
    layer's placed regions (tests and conversions)."""
    kb = k_stock.view(k_stock.shape[0], HEADS, -1).view(torch.uint8)
    vb = v_stock.view(v_stock.shape[0], HEADS, -1).view(torch.uint8)
    cn = np.asarray(codes, dtype=np.int64)
    xn = x_table.cpu().numpy()[cn]
    dev = layer.k.device
    K, V = layer.k, layer.v
    idx = lambda o, w: (
        torch.from_numpy(o).to(dev)[:, None] + torch.arange(w, device=dev)[None, :]
    ).reshape(-1)
    for h in range(HEADS):
        kd, ks, v0, v1 = placed_offsets(cn, xn, h)
        K[idx(kd, 16384)] = kb[:, h, :16384].reshape(-1)
        K[idx(ks, 2048)] = kb[:, h, 16384:18432].reshape(-1)
        V[idx(v0, 16384)] = vb[:, h, :16384].reshape(-1)
        V[idx(v1, 16384)] = vb[:, h, 16384:].reshape(-1)


def placed_to_stock(layer: KvIodLayer, codes, x_table, k_like, v_like):
    """Gather the placed bytes of blocks `codes` back into stock-shaped pages (tests)."""
    cn = np.asarray(codes, dtype=np.int64)
    xn = x_table.cpu().numpy()[cn]
    dev = layer.k.device
    kb = torch.empty((len(cn), HEADS, 18432), dtype=torch.uint8, device=dev)
    vb = torch.empty((len(cn), HEADS, 32768), dtype=torch.uint8, device=dev)
    idx = lambda o, w: (
        torch.from_numpy(o).to(dev)[:, None] + torch.arange(w, device=dev)[None, :]
    )
    for h in range(HEADS):
        kd, ks, v0, v1 = placed_offsets(cn, xn, h)
        kb[:, h, :16384] = layer.k[idx(kd, 16384)]
        kb[:, h, 16384:] = layer.k[idx(ks, 2048)]
        vb[:, h, :16384] = layer.v[idx(v0, 16384)]
        vb[:, h, 16384:] = layer.v[idx(v1, 16384)]
    return kb, vb


def resolve_block_tables(
    block_table: torch.Tensor, x_tables: torch.Tensor, out: torch.Tensor | None = None
):
    """Per-layer block tables with each code's x byte in bits 29-30 (code | x_tables[l, code] << 29), for
    unified_attention(kv_loc=(..., resolved=True)): the kernels then skip a dependent x lookup a tile.
    block_table [B, W] int32 codes, x_tables [layers, codes] uint8 -> [layers, B, W] int32.
    """
    r = x_tables[:, block_table.long()].to(torch.int32) << 29
    if out is None:
        return r.bitwise_or_(block_table)
    torch.bitwise_or(r, block_table, out=out)
    return out

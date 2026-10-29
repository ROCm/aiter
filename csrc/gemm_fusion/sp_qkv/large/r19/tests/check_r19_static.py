#!/usr/bin/env python3
"""Independent static checks for the ABI-7 large tree csrc/gemm_fusion/sp_qkv/large/r19 (source only; no compiler, no GPU).

r19 = the r13/large kernel extended to local M <= 16384 (LG_MAX_CHUNKS 64). The device code carries the compile-time proof layers
L1/L2 (device/lg_r6x.hpp) and the ABI/control-window static asserts (interface/lg_comm_abi_v6.h); this file is the INDEPENDENT L3
mirror of both, plus the tag-packing and control-layout checks. Nothing here includes or executes the C++ code: every formula is
re-implemented from the header comments/constants, so a divergence between the two implementations is a failure here.

Checks (all CPU):
  A. header parse: ABI 7, LG_MAX_CHUNKS 64 / LG_CHUNK_ROWS 256, tag partition 18+14 bits, control offsets.
  B. control-window layout: every array/word range disjoint, inside [0, LGC_BYTES), and the protocol words (CREDIT 64, ARRIVAL 128) unmoved.
  C. tag packing: E = (STEP<<32)|((m_real-1)<<18)|(call_id & 0x3FFFF) is injective on the (m_real, call_id) domain and its low word is 32 bits.
  D. item-order proofs mirrored: r6x_order_ok / r7_order_ok / r9_order_ok for EVERY chunks 1..LG_MAX_CHUNKS x every legal plan, dense and indexer.
  E. r13 live-row law for EVERY legal plan x every m_real 1..chunks*256 (pad-M tail: live rows sum to m_real and form a contiguous prefix).
  F. address model (r13 put/self): for every (rank, source) the four senders tile R = [4*m_real, F] exactly once and every source stays inside its S slab.
  G. SOURCE_SHA256.txt: --write-manifest regenerates it (sorted, sha256sum format); the default verifies it when present.
  H. informational geometry table (items / grid / 256-block rounds for the new m range) printed for the design note.

Run:  python tests/check_r19_static.py                         (checks + manifest verify)
      python tests/check_r19_static.py --write-manifest        (regenerate SOURCE_SHA256.txt, then checks pass)
"""

from __future__ import annotations

import hashlib
import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # large/r19
HEADER = os.path.join(ROOT, "interface", "lg_comm_abi_v6.h")

# ---- constants mirrored from interface/lg_comm_abi_v6.h / device/layout (kept literal; the header parse below cross-checks the macros) ----
CHUNK_ROWS = 256
NPD = 9  # LG_TILES_N_PER_DEST: (F - indexer*256)/256, F = 2304 dense / 2560 indexer -> 9 in both
NPEER = 3  # remote destinations of one rank
M_MAX = 16384
TAG_CALLID_BITS = 18
TAG_CALLID_MASK = 0x3FFFF
TAG_M_SHIFT = 18
LGC = {
    "EP": 0,
    "POISON": 8,
    "MAGIC": 16,
    "CREDIT": 64,
    "ARRIVAL": 128,
    "DONE": 512,
    "STATE": 1536,
    "LOCK": 2560,
    "FOLD_CNT": 2620,
    "REPORT": 2624,
    "POSTCNT": 2688,
    "CTAG": 3712,
    "ATAG": 3744,
    "QSTAMP": 4096,
    "XCCQ": 4352,
    "FIRSTPUT": 4416,
    "PSTAMP": 5120,
    "STAMP": 65536,
    "BYTES": 8192,
}
PLANS = ((1, 0), (2, 0), (4, 0), (2, 1), (4, 1), (3, 2))  # (NP, HG): n1 / n2 / n4 / n2h / n4h / n3q


def read_header():
    with open(HEADER, encoding="utf-8") as f:
        return f.read()


def parse_defines(text):
    """name -> literal int for the plain `#define NAME <int>[u]` macros (expressions are handled by name where needed)."""
    out = {}
    for m in re.finditer(r"^#define\s+([A-Z0-9_]+)\s+([0-9]+)u?\s*(?:/\*.*)?$", text, re.M):
        out[m.group(1)] = int(m.group(2))
    for m in re.finditer(r"^#define\s+([A-Z0-9_]+)\s+0x([0-9A-Fa-f]+)u?\s*(?:/\*.*)?$", text, re.M):
        out[m.group(1)] = int(m.group(2), 16)
    return out


# =============================== A. header parse ===============================
def check_header(text):
    ok = True

    def need(name, want):
        nonlocal ok
        m = re.search(r"^#define\s+%s\s+([0-9]+)u?" % re.escape(name), text, re.M)
        got = int(m.group(1)) if m else None
        if got != want:
            print(f"FAIL header: {name} = {got}, expected {want}")
            ok = False

    need("LG_COMM_ABI_VERSION", 7)
    need("LG_CHUNK_ROWS", CHUNK_ROWS)
    need("LG_MAX_CHUNKS", 64)
    need("LG_TAG_CALLID_BITS", TAG_CALLID_BITS)
    need("LG_TAG_M_SHIFT", TAG_M_SHIFT)
    m = re.search(r"^#define\s+LG_TAG_CALLID_MASK\s+0x([0-9A-Fa-f]+)u?", text, re.M)
    if not m or int(m.group(1), 16) != TAG_CALLID_MASK:
        print("FAIL header: LG_TAG_CALLID_MASK")
        ok = False
    d = parse_defines(text)
    for key, want in LGC.items():
        macro = "LGC_%s_OFF" % key if key != "BYTES" else "LGC_BYTES"
        if macro not in d:
            continue
        if d[macro] != want:
            print(f"FAIL header: {macro} = {d[macro]}, expected {want}")
            ok = False
    if not (64 * CHUNK_ROWS == M_MAX == (1 << (32 - TAG_M_SHIFT))):
        print("FAIL: LG_MAX_CHUNKS*LG_CHUNK_ROWS != 1<<(32-LG_TAG_M_SHIFT) != M_MAX")
        ok = False
    if TAG_M_SHIFT != TAG_CALLID_BITS or TAG_M_SHIFT + 14 != 32:
        print("FAIL: tag partition is not 18-bit call_id | 14-bit m_real")
        ok = False
    if "LgCommArgsV6" not in text or "LG_MREAL(a) ((a).m_real)" not in text:
        print("FAIL: the args struct / LG_MREAL(m_real) is missing")
        ok = False
    if re.search(r"^#define\s+LG_R13_ARGS", text, re.M):
        print("FAIL: the removed r13 LG_R13_ARGS knob is still referenced in the ABI header")
        ok = False
    print("PASS header (ABI 7, chunks 64, tag 18/14, args 360 B with m_real@352)" if ok else "FAILED header")
    return ok


# =============================== B. control layout ===============================
def check_layout():
    """Byte ranges of every C-window array/word; LGC_MAX_CHUNKS copies."""
    n = 64  # LG_MAX_CHUNKS
    ranges = {
        "EP(u64)": (LGC["EP"], 8),
        "POISON(u32)": (LGC["POISON"], 4),
        "MAGIC(u64)": (LGC["MAGIC"], 8),
        "CREDIT(u64[4])": (LGC["CREDIT"], 4 * 8),
        "ARRIVAL(u64[4])": (LGC["ARRIVAL"], 4 * 8),
        "DONE(u32[4][64])": (LGC["DONE"], 4 * n * 4),
        "STATE(u32[4][64])": (LGC["STATE"], 4 * n * 4),
        "LOCK(u32[4])": (LGC["LOCK"], 4 * 4),
        "FOLD_CNT(u32)": (LGC["FOLD_CNT"], 4),
        "REPORT(u64[8])": (LGC["REPORT"], 8 * 8),
        "POSTCNT(u32[4][64])": (LGC["POSTCNT"], 4 * n * 4),
        "CTAG(u64[4])": (LGC["CTAG"], 4 * 8),
        "ATAG(u64[4])": (LGC["ATAG"], 4 * 8),
        "QSTAMP(u64[3][8])": (LGC["QSTAMP"], 3 * 8 * 8),
        "XCCQ(u32[8])": (LGC["XCCQ"], 8 * 4),
        "FIRSTPUT(u64)": (LGC["FIRSTPUT"], 8),
        "PSTAMP(u64[3][16])": (LGC["PSTAMP"], 3 * 16 * 8),
    }
    ok = True
    spans = sorted((start, start + size, name) for name, (start, size) in ranges.items())
    for (s0, e0, n0), (s1, e1, n1) in zip(spans, spans[1:]):
        if s1 < e0:
            print(f"FAIL layout: {n0} [{s0},{e0}) overlaps {n1} [{s1},{e1})")
            ok = False
    for start, end, name in spans:
        if end > LGC["BYTES"]:
            print(f"FAIL layout: {name} ends at {end} > LGC_BYTES {LGC['BYTES']}")
            ok = False
    exp = {"DONE": 512, "STATE": 1536, "LOCK": 2560, "FOLD_CNT": 2620, "REPORT": 2624, "POSTCNT": 2688, "CTAG": 3712, "ATAG": 3744}
    for k, v in exp.items():
        if LGC[k] != v:
            print(f"FAIL layout: LGC_{k}_OFF = {LGC[k]}, expected {v}")
            ok = False
    if LGC["STATE"] != LGC["DONE"] + 4 * n * 4:
        print("FAIL layout: DONE does not end exactly at STATE (done[4][64] must not collide)")
        ok = False
    if LGC["LOCK"] != LGC["STATE"] + 4 * n * 4 or LGC["REPORT"] < LGC["LOCK"] + 64:
        print("FAIL layout: STATE/LOCK/REPORT partition")
        ok = False
    if LGC["POSTCNT"] != LGC["REPORT"] + 64 or LGC["CTAG"] != LGC["POSTCNT"] + 4 * n * 4:
        print("FAIL layout: POSTCNT/CTAG partition")
        ok = False
    if LGC["ATAG"] != LGC["CTAG"] + 32:
        print("FAIL layout: ATAG not packed after CTAG")
        ok = False
    if LGC["CREDIT"] != 64 or LGC["ARRIVAL"] != 128:
        print("FAIL layout: the r11..r13 CREDIT/ARRIVAL offsets moved")
        ok = False
    if LGC["QSTAMP"] < LGC["ATAG"] + 32:
        print("FAIL layout: the debug stamps overlap the tag words")
        ok = False
    for start, end, name in spans:
        print(f"  {name:22s} [{start:6d},{end:6d})  {end - start:5d} B")
    print("PASS control layout (arrays disjoint, protocol words unmoved)" if ok else "FAILED control layout")
    return ok


# =============================== C. tag packing ===============================
def tag_low(m_real, call_id):
    return (((m_real - 1) & 0x3FFF) << TAG_M_SHIFT) | (call_id & TAG_CALLID_MASK)


def check_tag():
    ok = True
    mset = {tag_low(m, 0) for m in range(1, M_MAX + 1)}
    cset = {tag_low(1, c) for c in range(TAG_CALLID_MASK + 1)}
    if len(mset) != M_MAX:
        print("FAIL tag: m_real is not injective on 1..16384")
        ok = False
    if len(cset) != TAG_CALLID_MASK + 1:
        print("FAIL tag: call_id is not injective on 18 bits")
        ok = False
    for m in (1, 2, 255, 256, 257, 8192, 8193, 16384):
        for c in (0, 1, 2, TAG_CALLID_MASK - 1, TAG_CALLID_MASK):
            lo = tag_low(m, c)
            if lo >= 1 << 32:
                print(f"FAIL tag: low word overflow m={m} c={c}")
                ok = False
            if (lo & TAG_CALLID_MASK) != (c & TAG_CALLID_MASK):
                print(f"FAIL tag: call_id field wrong m={m} c={c}")
                ok = False
            if ((lo >> TAG_M_SHIFT) & 0x3FFF) != ((m - 1) & 0x3FFF):
                print(f"FAIL tag: m_real field wrong m={m} c={c}")
                ok = False
    if tag_low(M_MAX, TAG_CALLID_MASK) != 0xFFFFFFFF:
        print("FAIL tag: the maximal (m_real, call_id) does not fill the 32-bit tag word")
        ok = False
    print("PASS tag packing (E = (STEP<<32)|((m_real-1)<<18)|call_id[17:0]; injective over 1..16384 x 18 bits)" if ok else "FAILED tag packing")
    return ok


# =============================== D. item-order mirrors ===============================
def fieldmap():
    """mirror of quad_fp8_fieldmap.hpp (dense Q|K|V half-tile map: 72 half tiles = 4 dest x 9 tiles x 2 halves)."""
    fields = ((0, 0, 2048), (8192, 2048, 128), (8704, 2176, 128))
    kseg_per_dest = 18

    def dest(hv):
        return hv // kseg_per_dest

    def packed(hv):
        return (hv % kseg_per_dest) * 128

    def field(hv):
        p = packed(hv)
        return 0 if p < fields[1][1] else (1 if p < fields[2][1] else 2)

    def raw(hv):
        f = fields[field(hv)]
        return f[0] + dest(hv) * f[2] + (packed(hv) - f[1])

    return dest, packed, raw


HT_DEST, HT_PACKED, HT_RAW = fieldmap()


def constants(indexer):
    items = NPD + (2 if indexer else 0)  # LG_ITEMS_PER_GROUP
    items_chunk = 4 * NPD + (5 if indexer else 0)  # LG_X_ITEMS_PER_CHUNK
    xtra = 5 if indexer else 0
    return items, items_chunk, xtra


def r6x_group_items(chunks, np, items):
    return (chunks // np) * items


def grid_items_x(chunks, items_chunk):
    return items_chunk * chunks


def select_item_r6x(t, chunks, np, indexer):
    items, _items_chunk, xtra = constants(indexer)
    cpp = chunks // np
    per_g = cpp * (xtra + NPEER * NPD)
    rn = per_g * np
    if t < rn:
        g, w = divmod(t, per_g)
        if indexer and w < cpp * xtra:
            c, k = divmod(w, xtra)
            c += g * cpp
            if k == 0:
                return ("IK", c, 0, 0, 0)
            return ("IQ", c, 0, k - 1, 0)
        v = w - cpp * xtra
        s = cpp * NPD
        slot, u = divmod(v, s)
        return ("N", g * cpp + u // NPD, slot, u % NPD, 0)
    s = t - rn
    return ("N", s // NPD, NPEER, s % NPD, 0)


def r6x_order_ok(chunks, np, indexer):
    items, items_chunk, xtra = constants(indexer)
    seen_n = set()
    seen_q = set()
    seen_k = set()
    arr = [[0] * np for _ in range(4)]
    raw_cnt = [0] * 80
    pk_cnt = [[0] * 20 for _ in range(4)]
    cpp = chunks // np
    for t in range(grid_items_x(chunks, items_chunk)):
        kind, xchunk, dslot, nloc, nsel = select_item_r6x(t, chunks, np, indexer)
        if xchunk < 0 or xchunk >= chunks:
            return False
        g = xchunk // cpp
        if kind == "N":
            if dslot < 0 or dslot > NPEER or nloc < 0 or nloc >= NPD:
                return False
            dst = dslot + 1 if dslot < NPEER else 0
            if (xchunk, dst, nloc) in seen_n:
                return False
            seen_n.add((xchunk, dst, nloc))
            for hn in (0, 1):
                hv = (dst * NPD + nloc) * 2 + hn
                if HT_DEST(hv) != dst:
                    return False
                if xchunk == 0:
                    raw_cnt[HT_RAW(hv) // 128] += 1
                    pk_cnt[HT_DEST(hv)][HT_PACKED(hv) // 128] += 1
            if dst != 0:
                arr[dst][g] += 1
        elif kind == "IQ":
            if not indexer or nloc < 0 or nloc > 3 or (xchunk, nloc) in seen_q:
                return False
            seen_q.add((xchunk, nloc))
            if xchunk == 0:
                raw_cnt[9216 // 128 + nloc] += 1
                pk_cnt[nloc][2304 // 128] += 1
            if nloc != 0:
                arr[nloc][g] += 1
        elif kind == "IK":
            if not indexer or xchunk in seen_k:
                return False
            seen_k.add(xchunk)
            if xchunk == 0:
                raw_cnt[9728 // 128] += 1
                for d in range(4):
                    pk_cnt[d][2432 // 128] += 1
            for d in (1, 2, 3):
                arr[d][g] += 1
        else:
            return False
    nseg, npk = (77, 20) if indexer else (72, 18)
    if any(raw_cnt[i] != 1 for i in range(nseg)):
        return False
    if any(pk_cnt[d][p] != 1 for d in range(4) for p in range(npk)):
        return False
    for d in (1, 2, 3):
        for g in range(np):
            if arr[d][g] != r6x_group_items(chunks, np, items):
                return False
    return True


def r7_want(chunks, np, g, items):
    return (chunks // np) * (items + (NPD if g == 0 else 0))


def grid_items_h(chunks, np, hg, items_chunk):
    return items_chunk * chunks + (NPEER * NPD * (chunks // np) if hg else 0)


def select_item_r7(t, chunks, np, indexer):
    items, _items_chunk, xtra = constants(indexer)
    cpp = chunks // np
    g0 = cpp * (xtra + 2 * NPEER * NPD)
    gn = cpp * (xtra + NPEER * NPD)
    rn = g0 + (np - 1) * gn
    if t < rn:
        if t < g0:
            g, w = 0, t
        else:
            g, w = 1 + (t - g0) // gn, (t - g0) % gn
        if indexer and w < cpp * xtra:
            c, k = divmod(w, xtra)
            c += g * cpp
            if k == 0:
                return ("HK", c, 0, 0, 0)
            return ("HQ", c, 0, k - 1, 0)
        v = w - cpp * xtra
        if g == 0:
            s = cpp * NPD * 2
            slot, u = divmod(v, s)
            u2 = u // 2
            return ("HN", g * cpp + u2 // NPD, slot, u2 % NPD, u % 2)
        s = cpp * NPD
        slot, u = divmod(v, s)
        return ("N", g * cpp + u // NPD, slot, u % NPD, 0)
    s = t - rn
    return ("N", s // NPD, NPEER, s % NPD, 0)


def r7_order_ok(chunks, np, indexer):
    items, items_chunk, _xtra = constants(indexer)
    seen_n = set()
    seen_q = set()
    seen_k = set()
    arr = [[0] * np for _ in range(4)]
    raw_cnt = [0] * 80
    pk_cnt = [[0] * 20 for _ in range(4)]
    cpp = chunks // np
    for t in range(grid_items_h(chunks, np, 1, items_chunk)):
        kind, xchunk, dslot, nloc, nsel = select_item_r7(t, chunks, np, indexer)
        if xchunk < 0 or xchunk >= chunks:
            return False
        g = xchunk // cpp
        if kind in ("N", "HN"):
            if dslot < 0 or dslot > NPEER or nloc < 0 or nloc >= NPD:
                return False
            if kind == "HN" and (g != 0 or dslot == NPEER or nsel < 0 or nsel > 1):
                return False
            if kind == "N" and g == 0 and dslot != NPEER:
                return False
            dst = dslot + 1 if dslot < NPEER else 0
            for hn in (0, 1):
                if kind == "HN" and hn != nsel:
                    continue
                if (xchunk, dst, nloc, hn) in seen_n:
                    return False
                seen_n.add((xchunk, dst, nloc, hn))
                hv = (dst * NPD + nloc) * 2 + hn
                raw_cnt[HT_RAW(hv) // 128] += 1
                pk_cnt[HT_DEST(hv)][HT_PACKED(hv) // 128] += 1
                if HT_DEST(hv) != dst:
                    return False
            if dst != 0:
                arr[dst][g] += 1
        elif kind == "HQ":
            if not indexer or nloc < 0 or nloc > 3 or (xchunk, nloc) in seen_q:
                return False
            seen_q.add((xchunk, nloc))
            raw_cnt[9216 // 128 + nloc] += 1
            pk_cnt[nloc][2304 // 128] += 1
            if nloc != 0:
                arr[nloc][g] += 1
        elif kind == "HK":
            if not indexer or xchunk in seen_k:
                return False
            seen_k.add(xchunk)
            raw_cnt[9728 // 128] += 1
            for d in range(4):
                pk_cnt[d][2432 // 128] += 1
            for d in (1, 2, 3):
                arr[d][g] += 1
        else:
            return False
    nseg, npk = (77, 20) if indexer else (72, 18)
    if any(raw_cnt[i] != chunks for i in range(nseg)):
        return False
    if any(pk_cnt[d][p] != chunks for d in range(4) for p in range(npk)):
        return False
    for d in (1, 2, 3):
        for g in range(np):
            if arr[d][g] != r7_want(chunks, np, g, items):
                return False
    return True


# =============================== E. live rows ===============================
def g_first(NP, HG, g, chunks):
    return [0, 2, 3][g] if HG == 2 else g * (chunks // NP)


def g_n(NP, HG, g, chunks):
    return [2, 1, chunks - 3][g] if HG == 2 else chunks // NP


def live_rows(m, c0, cpp):
    lo, hi = c0 * CHUNK_ROWS, (c0 + cpp) * CHUNK_ROWS
    e = min(m, hi)
    return e - lo if e > lo else 0


def live_ok(NP, HG, chunks):
    for m in range(1, chunks * CHUNK_ROWS + 1):
        s = 0
        tail = False
        for g in range(NP):
            c0, cpp = g_first(NP, HG, g, chunks), g_n(NP, HG, g, chunks)
            lr = live_rows(m, c0, cpp)
            if tail and lr != 0:
                return False
            if lr != cpp * CHUNK_ROWS:
                tail = True
            s += lr
        if s != m:
            return False
    return True


def legal_plans():
    """every (chunks, NP, HG) the host can launch (plan-shape legality of lg_comm_launch)."""
    for chunks in range(1, 65):
        for NP, HG in PLANS:
            if HG == 2:
                if NP == 3 and chunks == 8:
                    yield chunks, NP, HG
                continue
            if chunks % NP == 0:
                yield chunks, NP, HG


def r9_want(chunks, g):
    R9_QC, R9_G1 = 2, 1
    return R9_QC * NPD * 4 if g == 0 else ([R9_QC, R9_G1, chunks - R9_QC - R9_G1][g]) * NPD


def grid_items_q(chunks):
    R9_QC = 2
    return NPEER * NPD * (R9_QC * 4 + (chunks - R9_QC)) + NPD * chunks


def select_item_r9(t, chunks):
    R9_QC, R9_G1 = 2, 1
    S0, S1 = R9_QC * NPD * 4, R9_G1 * NPD
    g0, g1 = NPEER * S0, NPEER * S1
    S2 = (chunks - R9_QC - R9_G1) * NPD
    Rn = g0 + g1 + NPEER * S2
    if t < g0:
        slot, u = divmod(t, S0)
        q, u2 = u % 4, u // 4
        return ("QN", u2 // NPD, slot, u2 % NPD, q)
    if t < g0 + g1:
        v = t - g0
        slot, u = divmod(v, S1)
        return ("N", R9_QC + u // NPD, slot, u % NPD, 0)
    if t < Rn:
        v = t - g0 - g1
        slot, u = divmod(v, S2)
        return ("N", R9_QC + R9_G1 + u // NPD, slot, u % NPD, 0)
    s = t - Rn
    return ("N", s // NPD, NPEER, s % NPD, 0)


def r9_order_ok(chunks):
    R9_QC, R9_G1 = 2, 1
    seen_t = set()
    cov = set()
    arr = [[0] * 3 for _ in range(4)]
    for t in range(grid_items_q(chunks)):
        kind, xchunk, dslot, nloc, nsel = select_item_r9(t, chunks)
        if xchunk < 0 or xchunk >= chunks or dslot < 0 or dslot > NPEER or nloc < 0 or nloc >= NPD:
            return False
        dst = dslot + 1 if dslot < NPEER else 0
        g = 0 if xchunk < R9_QC else (1 if xchunk < R9_QC + R9_G1 else 2)
        if kind == "QN":
            if g != 0 or dslot == NPEER or nsel < 0 or nsel > 3:
                return False
            hn, hm = nsel & 1, nsel >> 1
            if (xchunk, dst, nloc, hn, hm) in cov:
                return False
            cov.add((xchunk, dst, nloc, hn, hm))
        elif kind == "N":
            if g == 0 and dslot != NPEER:
                return False
            if (xchunk, dst, nloc) in seen_t:
                return False
            seen_t.add((xchunk, dst, nloc))
            for hn in (0, 1):
                for hm in (0, 1):
                    if (xchunk, dst, nloc, hn, hm) in cov:
                        return False
                    cov.add((xchunk, dst, nloc, hn, hm))
        else:
            return False
        if dst != 0:
            arr[dst][g] += 1
    for c in range(chunks):
        for d in range(4):
            for nn in range(NPD):
                for hn in (0, 1):
                    for hm in (0, 1):
                        if (c, d, nn, hn, hm) not in cov:
                            return False
    for d in (1, 2, 3):
        for g in range(3):
            if arr[d][g] != r9_want(chunks, g):
                return False
    return True


def check_order_and_live():
    ok = True
    n_plans = 0
    for indexer in (0, 1):
        for chunks, NP, HG in legal_plans():
            n_plans += 1
            if HG == 0:
                good = r6x_order_ok(chunks, NP, indexer)
            elif HG == 1:
                good = r7_order_ok(chunks, NP, indexer)
            else:
                if indexer:
                    continue  # the host refuses n3q on the indexer build (-103): not a launchable shape
                good = r9_order_ok(chunks)
            if not good:
                print(f"FAIL order: chunks={chunks} NP={NP} HG={HG} indexer={indexer}")
                ok = False
    print(f"PASS item order (r6x/r7/r9 mirrors) for {n_plans} (chunks, plan) pairs x both builds" if ok else "FAILED item order")
    ok2 = True
    n_live = 0
    for chunks, NP, HG in legal_plans():
        n_live += 1
        if not live_ok(NP, HG, chunks):
            print(f"FAIL live rows: chunks={chunks} NP={NP} HG={HG}")
            ok2 = False
    print(f"PASS live rows for {n_live} (chunks, plan) pairs x every m_real 1..chunks*256" if ok2 else "FAILED live rows")
    return ok and ok2


# =============================== F. address model ===============================
def check_address_model():
    ok = True
    for F in (2304, 2560):
        for m_real, srows in itertools.product(
            (1, 17, 256, 257, 300, 384, 450, 512, 585, 700, 768, 1024, 2049, 8192, 8193, 16000, 16383, 16384),
            (None, 16384),
        ):
            if srows is None:
                srows = ((m_real + 255) // 256) * 256
            if srows < m_real or srows > 16384:
                continue
            for dst_rank in range(4):
                spans = []
                for src in range(4):
                    dst_off = src * 2 * m_real * F
                    nbytes = 2 * m_real * F
                    src_off = src * 2 * srows * F if src == dst_rank else dst_rank * 2 * srows * F
                    lo, hi = dst_rank * 2 * srows * F, (dst_rank + 1) * 2 * srows * F
                    if not (lo <= src_off and src_off + nbytes <= hi):
                        print(f"FAIL address: source leaves slab F={F} m_real={m_real} srows={srows}")
                        ok = False
                    spans.append((dst_off, dst_off + nbytes))
                spans.sort()
                pos = 0
                for lo, hi in spans:
                    if lo != pos:
                        print(f"FAIL address: R gap/overlap F={F} m_real={m_real}")
                        ok = False
                    pos = hi
                if pos != 4 * 2 * m_real * F:
                    print(f"FAIL address: R not tiled exactly F={F} m_real={m_real}")
                    ok = False
    print("PASS address model (R tile [4*m_real, F] tiled exactly once, sources inside their S slabs)" if ok else "FAILED address model")
    return ok


# =============================== G. manifest ===============================
MANIFEST = os.path.join(ROOT, "SOURCE_SHA256.txt")


def manifest_files():
    out = []
    for base, _dirs, files in os.walk(ROOT):
        for f in files:
            full = os.path.join(base, f)
            rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
            if rel == "SOURCE_SHA256.txt" or "__pycache__" in rel or rel.endswith(".pyc"):
                continue
            out.append(rel)
    return sorted(out)


def write_manifest():
    lines = []
    for rel in manifest_files():
        with open(os.path.join(ROOT, rel), "rb") as f:
            lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {rel}")
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"WROTE manifest {MANIFEST} ({len(lines)} files)")


def check_manifest():
    if not os.path.exists(MANIFEST):
        print("FAIL manifest: SOURCE_SHA256.txt is missing (run --write-manifest)")
        return False
    ok = True
    listed = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            h, rel = line.split("  ", 1)
            listed[rel] = h
    files = set(manifest_files())
    for rel in sorted(files - set(listed)):
        print(f"FAIL manifest: missing entry {rel}")
        ok = False
    for rel in sorted(set(listed) - files):
        print(f"FAIL manifest: stale entry {rel}")
        ok = False
    for rel in sorted(files & set(listed)):
        with open(os.path.join(ROOT, rel), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != listed[rel]:
            print(f"FAIL manifest: hash mismatch {rel}")
            ok = False
    print(f"PASS manifest ({len(listed)} files)" if ok else "FAILED manifest")
    return ok


# =============================== H. geometry ===============================
def print_geometry():
    print("geometry (informational): r13-style item count / grid / 256-block rounds")
    print(f"  {'M':>6} {'chunks':>6} {'dense items':>11} {'rounds':>6} {'index items':>11} {'rounds':>6}")
    for M in (1024, 2048, 4096, 8192, 10240, 12288, 16384):
        c = M // CHUNK_ROWS
        di, ii = 36 * c, 41 * c
        print(f"  {M:>6} {c:>6} {di:>11} {-(-(di + 1) // 256):>6} {ii:>11} {-(-(ii + 1) // 256):>6}")


def main(argv):
    if "--write-manifest" in argv:
        write_manifest()
    results = [
        check_header(read_header()),
        check_layout(),
        check_tag(),
        check_order_and_live(),
        check_address_model(),
        check_manifest(),
    ]
    print_geometry()
    allok = all(results)
    print("R19 STATIC", "ALL PASS" if allok else "FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

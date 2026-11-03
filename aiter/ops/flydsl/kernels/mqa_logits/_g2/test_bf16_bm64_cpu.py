# SPDX-License-Identifier: MIT
"""Independent, stdlib-only BM64 bf16 layout and source-control-flow checks.

M16 means FOUR query rows x FOUR heads, not sixteen query rows. BM64 has
four M16 tiles per wave, four waves, and two sequential pairs per wave. A
single 17,408-byte padded K tile is produced by eight gathers (two per wave),
then consumed by both pairs without being restaged.

Opaque uint16 payloads are compared byte-for-byte for every lane, K chunk,
N tile, pair, and wave. Invalid pages copy physical page zero with c%4 retained;
visibility and column/store masks never zero the staged payload. Q mapping and
C/head reduction ownership are checked independently. The production kernel
AST is also replayed with scalar CPU helpers and SYMBOLIC scores, testing its
actual branches, pair/tail indexing, masked stores and metadata writes. This
is NOT a floating-point WMMA, lowering/ISA, device correctness or speed proof.
No aiter, torch, FlyDSL import, compilation, device discovery or GPU work.

Run: python -B aiter/ops/flydsl/kernels/mqa_logits/_g2/test_bf16_bm64_cpu.py -v
"""
from __future__ import annotations

import sys as _n64_sys, pathlib as _n64_path  # ROUND-NATIVE64 seam support
_n64_sys.path.insert(0, str(_n64_path.Path(__file__).resolve().parent))
from native64_seam import ensure_i32  # noqa: E402

import ast
from collections import Counter
import copy
from dataclasses import dataclass
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace
import unittest


PAGE, HEADS, DIM = 4, 4, 128
WAVES, LANES, MROWS, MTILES, COLS = 4, 32, 4, 4, 64
ROW_BYTES, PAD_ROW_BYTES, LDS_BYTES = 256, 272, 17408
PAD = bytes.fromhex("a55a3cc396690ff05aa5c33c6996f00f")
SOURCE = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"


class Scalar(int):
    """Tiny scalar select shim; no vector, arithmetic WMMA, or runtime import."""
    def select(self, yes, no):
        return yes if self else no

    def __add__(self, other): return Scalar(int(self) + int(other))
    def __radd__(self, other): return Scalar(int(other) + int(self))
    def __sub__(self, other): return Scalar(int(self) - int(other))
    def __rsub__(self, other): return Scalar(int(other) - int(self))
    def __mul__(self, other): return Scalar(int(self) * int(other))
    def __rmul__(self, other): return Scalar(int(other) * int(self))
    def __floordiv__(self, other): return Scalar(int(self) // int(other))
    def __mod__(self, other): return Scalar(int(self) % int(other))
    def __and__(self, other): return Scalar(bool(self) and bool(other))
    def __rand__(self, other): return Scalar(bool(other) and bool(self))
    def __or__(self, other): return Scalar(bool(self) or bool(other))
    def __ror__(self, other): return Scalar(bool(other) or bool(self))
    def __lt__(self, other): return Scalar(int(self) < int(other))
    def __le__(self, other): return Scalar(int(self) <= int(other))
    def __ge__(self, other): return Scalar(int(self) >= int(other))
    def __gt__(self, other): return Scalar(int(self) > int(other))
    def __eq__(self, other): return Scalar(int(self) == other)
    def __ne__(self, other): return Scalar(int(self) != other)
    __hash__ = int.__hash__


@dataclass(frozen=True)
class Gather:
    wave: int
    group: int
    offset: int
    rows: tuple[int, ...]


def stage_bytes(source, table, origin, order=tuple(range(8))):
    """Independent descriptor DWORD model. It does NOT call raw_fragment."""
    assert len(source) % (PAGE * ROW_BYTES) == 0 and table
    assert origin >= 0 and origin % COLS == 0 and sorted(order) == list(range(8))
    pages = len(source) // (PAGE * ROW_BYTES)
    scratch = bytearray(PAD * (LDS_BYTES // 16))
    writes = Counter()
    gathers = []
    for op in order:
        local = op * 8
        ids = []
        # A gather uses two scalar page lookups expanded to eight row indices.
        for page in range(2):
            logical = (origin + local) // PAGE + page
            raw = table[logical if logical < len(table) else 0]
            physical = raw if logical < len(table) and 0 <= raw < pages else 0
            ids.extend(physical * PAGE + row for row in range(PAGE))
        ids = struct.unpack("<8i", struct.pack("<8i", *ids))
        offset = local * PAD_ROW_BYTES
        gathers.append(Gather(op // 2, op % 2, offset, ids))
        # 128 bf16 elements = 64 DWORDs; skip four DWORDs after each row.
        for dw in range(8 * 64):
            src = ids[dw // 64] * ROW_BYTES + (dw % 64) * 4
            dst = offset + (dw + (dw // 64) * 4) * 4
            assert 0 <= src <= len(source) - 4 and 0 <= dst <= LDS_BYTES - 4
            scratch[dst:dst + 4] = source[src:src + 4]
            writes.update(range(dst, dst + 4))
    return bytes(scratch), tuple(gathers), writes


def raw_fragment(source, tables, request, c, lane, step):
    """Independent global loads using scalar token addressing, not gather ids."""
    pages = len(source) // 1024
    req_ok = 0 <= request < len(tables)
    table = tables[request if req_ok else 0]
    logical_ok = c // 4 < len(table)
    physical = table[c // 4 if logical_ok else 0]
    page_ok = req_ok and logical_ok and 0 <= physical < pages
    safe = physical if page_ok else 0
    base = safe * 1024 + (c % 4) * 256 + (lane // 16) * 16 + step * 64
    assert 0 <= base and base + 48 <= len(source)
    return source[base:base + 16] + source[base + 32:base + 48], page_ok


def shared_fragment(scratch, nt, lane, step):
    base = (nt * 16 + lane % 16) * 272 + (lane // 16) * 16 + step * 64
    for offset in (base, base + 32):
        assert offset % 16 == 0 and 0 <= offset <= LDS_BYTES - 16
        assert offset % 272 + 16 <= 256  # Never read the padded bytes.
    return scratch[base:base + 16] + scratch[base + 32:base + 48]


def q_fragment(source, t0, rows, lane, step):
    row = min(t0 + (lane % 16) // 4, rows - 1)
    head = lane % 4
    base = row * 1024 + head * 256 + (lane // 16) * 16 + step * 64
    assert base % 16 == 0 and 0 <= base and base + 48 <= len(source)
    return source[base:base + 16] + source[base + 32:base + 48]


def query_reference(source, row, head, lane, step):
    """Element-coordinate reference, not the two-vector consumer address."""
    out = []
    for v in range(16):
        k = step * 32 + (v // 8) * 16 + (lane // 16) * 8 + v % 8
        offset = ((row * HEADS + head) * DIM + k) * 2
        out.append(source[offset:offset + 2])
    return b"".join(out)


def independent_guard(t2r, base, requests):
    subset = t2r[base:base + 64]
    return len(subset) == 64 and 0 <= subset[0] < requests and len(set(subset)) == 1


def _calls(root, name):
    return [n for n in ast.walk(root) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == name]


def _exec(nodes, namespace):
    nodes = copy.deepcopy(nodes)
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ensure_i32(namespace)
    exec(compile(module, str(SOURCE), "exec"), namespace)


def _int_expr(node, values):
    if isinstance(node, ast.Constant) and type(node.value) is int: return node.value
    if isinstance(node, ast.Name): return values[node.id]
    if isinstance(node, ast.BinOp):
        a, b = _int_expr(node.left, values), _int_expr(node.right, values)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.FloorDiv): return a // b
        if isinstance(node.op, ast.Mod): return a % b
    raise AssertionError("Unsupported scalar layout expression: " + ast.unparse(node))


class SourceReplay:
    """Execute real kernel_r64 and real _pair_values/_pass_masks source AST.

    Existing metadata/pass-plan/Q/tile arithmetic is replaced by independently
    specified CPU helpers. Score ids distinguish row, physical page and token;
    they deliberately make no claim about floating-point arithmetic.
    """
    def __init__(self, tree):
        self.functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        build = self.functions["_build_wmma_kernel"]
        self.helpers = {n.name: n for n in build.body if isinstance(n, ast.FunctionDef)}
        self.constants = dict(PAGE_SIZE=4, NUM_HEADS=4, HEAD_DIM=128,
                              WMMA_WAVE=32, WMMA_WAVES=4, WMMA_THREADS=128,
                              WMMA_ROWS_PER_WAVE=4, WMMA_N=16, WMMA_K=32,
                              WMMA_K_STEPS=4, WMMA_N_TILES=4, WMMA_PASSES=4,
                              WMMA_COLS=64, R16_TDM_COLS=32, _BF16_R64=False,
                              TDM_K_ROW_BYTES=272, TDM_K_TILE_BYTES=17408,
                              # Round PERSIST wrapper switch (default off).
                              TDM_PERSIST=False, TDM_PERSIST_CHUNKS=4)

    def namespace(self):
        return dict(self.constants, range_constexpr=range, _imin=min, _imax=max)

    def guard(self, t2r, base, requests):
        accesses = []
        def load(data, offset):
            assert offset % 4 == 0 and 0 <= offset < len(data) * 4
            accesses.append(int(offset))
            return Scalar(data[offset // 4])
        ns = self.namespace()
        ns["guard_load"] = load
        _exec([self.helpers["_r64_guard"]], ns)
        value = ns["_r64_guard"](t2r, base, len(t2r), requests)
        return bool(value), accesses

    def stage(self, t2r, tables, rblk, col0, pages):
        """Execute actual unchanged producer with rblk*2 adaptation."""
        records, events, loads = [], [], []
        ns = self.namespace()
        def load(data, offset):
            assert offset % 4 == 0 and 0 <= offset < len(data) * 4
            loads.append(("request" if data is t2r else "table", int(offset)))
            return Scalar(data[offset // 4])
        flat = tuple(value for table in tables for value in table)
        for wave in range(4):
            ns["fx"] = SimpleNamespace(Int32=Scalar, rocdl=SimpleNamespace(
                wave_id=lambda w=wave: w,
                tdm_ops=SimpleNamespace(tensor_wait=lambda count: events.append((wave, "wait", count)))),
                gpu=SimpleNamespace(barrier=lambda: events.append((wave, "barrier"))))
            ns["_ld_uniform_i32"] = load
            def gather(kv, ptr, ids, n, off):
                records.append(Gather(wave, len(records) % 2, int(off), tuple(map(int, ids))))
            ns["_tdm_k_gather"] = gather
            _exec([self.helpers["_pair_stage_tdm"]], ns)
            ns["_pair_stage_tdm"](None, None, flat, t2r, rblk * 2, col0,
                                  len(tables[0]), pages)
        return tuple(records), events, loads

    def replay(self, t2r, qpos, contexts, tables, pages, columns, tdm=True):
        rows = len(t2r)
        output, counts, vis, vis_counts, rst_counts = {}, Counter(), {}, Counter(), Counter()
        stage_calls, pair_calls, raw_calls, q_calls = [], [], [], []
        flat = tuple(v for table in tables for v in table)
        ns = self.namespace()
        ns.update(q_row_bytes=1024, page_bytes=1024, _NEG_INF=-math.inf)
        def plane(data, size):
            if isinstance(data, tuple): assert len(data) * 4 == size
            return data
        ns.update(_plane=plane, _plane_bytes=lambda *factors: math.prod(factors))
        def load(data, offset):
            assert offset % 4 == 0 and 0 <= offset < len(data) * 4
            return Scalar(data[offset // 4])
        ns["guard_load"] = load
        def rows_meta(t, p, cl, t0, nrows, nreq):
            metas = []
            for i in range(4):
                row, safe = int(t0) + i, min(int(t0) + i, nrows - 1)
                req = t[safe]
                req_ok = 0 <= req < nreq
                safe_req = req if req_ok else 0
                visible = max(min(p[safe] + 1, cl[safe_req]), 0) // 4 if req_ok else 0
                metas.append(SimpleNamespace(row=row, row_ok=row < nrows, req=req,
                             req_ok=Scalar(req_ok), safe_req=safe_req, vis=visible))
            return metas
        ns["_rows_meta"] = rows_meta
        def plans(meta, nrows, kgrp):
            out = []
            for p, mp in enumerate(meta):
                active = mp.row_ok and not any(m.row_ok and m.req == mp.req for m in meta[:p])
                own = [Scalar(m.row_ok and m.req == mp.req) for m in meta]
                a, b = 2 * int(kgrp), 2 * int(kgrp) + 1
                out.append(SimpleNamespace(active=active, vmax=max(m.vis if own[i] else 0
                    for i, m in enumerate(meta)), safe_req=mp.safe_req, req_ok=mp.req_ok,
                    own_a=own[a], own_b=own[b], vis_a=meta[a].vis, vis_b=meta[b].vis, kgrp=kgrp))
            return out
        ns["_pass_plan"] = plans
        ns["_pick_vis"] = lambda lane, meta: meta[min(int(lane), 3)].vis
        ns["_lane_cols"] = lambda origin, n: [origin + nt * 16 + n for nt in range(4)]
        def offsets(t0, kgrp, cols, nrows, ncols):
            return [Scalar(min(min(int(t0) + 2 * int(kgrp) + rb, nrows - 1) * ncols + c,
                               nrows * ncols - 1) * 4) for c in cols for rb in range(2)]
        ns["_store_offsets"] = offsets
        def qfrags(q, t0, nrows, n_in, kgrp):
            assert 0 <= min(t0 + n_in // 4, nrows - 1) < nrows
            q_calls.append((by, chunk, wave, lane, int(t0)))
            return int(t0)
        ns["_q_frags"] = qfrags
        ns["pair_storage"] = lambda kv, ptr: "shared" if tdm else kv
        def stage(kv, ptr, pt, t, adjusted, col0, width, n):
            stage_calls.append((by, chunk, wave, lane, int(adjusted), int(col0), ptr))
        ns["pair_stage"] = stage
        def score(row, c, pp):
            physical = tables[pp.safe_req][c // 4 if c // 4 < len(tables[0]) else 0]
            good = bool(pp.req_ok) and c // 4 < len(tables[0]) and 0 <= physical < pages
            # Symbolic row+K-token identity, not a floating-point score.
            return Scalar(good), 1 + row * 65536 + (physical if good else 0) * 4 + c % 4
        def tile_pair(atom, kv, pt, qa, qb, pp, c, kg, width, n):
            pair_calls.append((by, chunk, wave, lane, qa, qb, int(c), kv))
            page_ok, unused = score(qa, c, pp)
            return page_ok, [score(base + 2 * int(kg) + rb, c, pp)[1]
                             for base in (qa, qb) for rb in range(2)]
        ns["_tile_pair"] = tile_pair
        def pass_values(atom, kv, pt, qf, pp, cols, offs, ncols, width, n):
            raw_calls.append((by, chunk, wave, lane, qf, pp.safe_req, kv))
            result = []
            for nt, c in enumerate(cols):
                for rb in range(2):
                    row = qf + 2 * int(pp.kgrp) + rb
                    good, value = score(row, c, pp)
                    own, visible = (pp.own_a, pp.vis_a) if rb == 0 else (pp.own_b, pp.vis_b)
                    result.append((own & (c < ncols), offs[2 * nt + rb],
                                   value if good and c < visible else -math.inf))
            return result
        ns["_pass_values"] = pass_values
        def store(target, offset, value):
            assert offset % 4 == 0
            if target == "logits":
                assert 0 <= offset <= rows * columns * 4 - 4
                index = int(offset) // 4
                output[index] = value
                counts[index] += 1
            else:
                assert 0 <= offset <= rows * 4 - 4
                row = int(offset) // 4
                if target == "visible": vis[row] = value; vis_counts[row] += 1
                elif target == "starts": assert value == 0; rst_counts[row] += 1
                else: raise AssertionError(target)
        ns.update(_st_i32=store, _st_f32=store)
        _exec([self.helpers[n] for n in ("_r64_guard", "_pair_values", "_pass_masks", "kernel_r64")], ns)
        for by in range((rows + 63) // 64):
            for chunk in range((columns + 63) // 64):
                for wave in range(4):
                    for lane in range(32):
                        fx = SimpleNamespace(Int32=Scalar, Float32=lambda value: value, BFloat16=None,
                            thread_idx=SimpleNamespace(x=wave * 32 + lane),
                            block_idx=SimpleNamespace(x=chunk, y=by),
                            make_mma_atom=lambda atom: None,
                            rocdl=SimpleNamespace(WMMA=lambda *args: None))
                        ns["fx"] = fx
                        ns["kernel_r64"]("query", "keys", flat, t2r, qpos, contexts,
                                             "logits", "visible", "starts", Scalar(rows),
                                             Scalar(columns), Scalar(pages), Scalar(len(tables)),
                                             Scalar(len(tables[0])))
        return SimpleNamespace(output=output, counts=counts, vis=vis, vis_counts=vis_counts,
                               rst_counts=rst_counts, stages=stage_calls, pairs=pair_calls,
                               raw=raw_calls, queries=q_calls)


class BM64BytesCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patterns = list(range(65536))
        random.Random(0xB164).shuffle(patterns)
        cls.source = struct.pack("<65536H", *patterns)
        cls.pages = len(cls.source) // 1024

    def assert_tile(self, table, origin=0, order=tuple(range(8))):
        scratch, gathers, writes = stage_bytes(self.source, table, origin, order)
        self.assertEqual(len(scratch), 17408)
        self.assertEqual(Counter(g.wave for g in gathers), Counter({w: 2 for w in range(4)}))
        restored = bytearray(64 * 256)
        owners = Counter()
        # Both sequential pairs in all four waves read the SAME shared tile.
        for wave in range(4):
            for pair in range(2):
                for nt in range(4):
                    for step in range(4):
                        for lane in range(32):
                            c = origin + nt * 16 + lane % 16
                            raw, unused = raw_fragment(self.source, (table,), 0, c, lane, step)
                            actual = shared_fragment(scratch, nt, lane, step)
                            self.assertEqual(actual, raw, (wave, pair, nt, step, lane))
                            if wave == pair == 0:
                                for v in range(16):
                                    k = step * 32 + (v // 8) * 16 + lane // 16 * 8 + v % 8
                                    element = (c - origin) * 128 + k
                                    owners[element] += 1
                                    restored[element * 2:element * 2 + 2] = actual[v * 2:v * 2 + 2]
        self.assertEqual(owners, Counter({i: 1 for i in range(64 * 128)}))
        for row in range(64):
            self.assertEqual(scratch[row * 272:row * 272 + 256], restored[row * 256:(row + 1) * 256])
            self.assertEqual(scratch[row * 272 + 256:(row + 1) * 272], PAD)
            self.assertTrue(all(writes[b] == 1 for b in range(row * 272, row * 272 + 256)))
            self.assertTrue(all(writes[b] == 0 for b in range(row * 272 + 256, (row + 1) * 272)))
        self.assertEqual(len(writes), 16384, "K staged once, not once per pair/wave/M tile")
        self.assertEqual(sorted(g.offset for g in gathers), list(range(0, 17408, 2176)))
        return bytes(restored), gathers

    def test_every_bf16_bit_pattern_all_waves_pairs_lanes_and_kchunks(self):
        table = tuple(range(self.pages))
        restored = b"".join(self.assert_tile(table, origin)[0] for origin in range(0, 512, 64))
        self.assertEqual(restored, self.source)
        self.assertEqual(set(struct.unpack("<65536H", restored)), set(range(65536)))

    def test_invalid_pages_logical_width_tails_and_gather_schedules(self):
        for width in (1, 15, 17, 33):
            table = tuple((7, -1, self.pages, 0x7FFFFFFF, -0x80000000, 3)[i % 6]
                          for i in range(width))
            for origin in (0, 64, ((width * 4 + 63) // 64) * 64):
                with self.subTest(width=width, origin=origin):
                    actual, gathers = self.assert_tile(table, origin)
                    other, reverse, unused = stage_bytes(self.source, table, origin, tuple(reversed(range(8))))
                    expected, unused_g, unused_w = stage_bytes(self.source, table, origin)
                    self.assertEqual(other, expected, "producer scheduling must commute")
                    for g in gathers:
                        for i, rowid in enumerate(g.rows):
                            c = origin + g.offset // 272 + i
                            valid = c // 4 < width and 0 <= table[c // 4] < self.pages
                            if not valid: self.assertEqual(rowid, c % 4)

    def test_q_all64_rows_all4_heads_and_c_reduction_ownership(self):
        rows = 129
        q = struct.pack("<%dH" % (rows * 4 * 128),
                        *((i * 73 + i // 128 * 11) & 0xFFFF for i in range(rows * 4 * 128)))
        for block in (0, 1, 2):
            qowners, cowners = Counter(), Counter()
            for wave in range(4):
                for pair in range(2):
                    for within in range(2):
                        mi = 2 * pair + within
                        t0 = block * 64 + wave * 16 + mi * 4
                        for step in range(4):
                            for lane in range(32):
                                row = min(t0 + lane % 16 // 4, rows - 1)
                                head = lane % 4
                                self.assertEqual(q_fragment(q, t0, rows, lane, step),
                                                 query_reference(q, row, head, lane, step))
                                for v in range(16):
                                    k = step * 32 + (v // 8) * 16 + lane // 16 * 8 + v % 8
                                    if t0 + lane % 16 // 4 < rows: qowners[row, head, k] += 1
                        # C: eight accumulator slots per lane, four heads for
                        # each of the lane's two rows, all sixteen N columns.
                        for lane in range(32):
                            for v in range(8):
                                m = lane // 16 * 8 + v
                                row, head = t0 + m // 4, m % 4
                                if row < rows: cowners[row, head, lane % 16] += 1
            live = range(block * 64, min(block * 64 + 64, rows))
            self.assertEqual(qowners, Counter({(r, h, k): 1 for r in live for h in range(4) for k in range(128)}))
            self.assertEqual(cowners, Counter({(r, h, n): 1 for r in live for h in range(4) for n in range(16)}))


class BM64SourceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.replay = SourceReplay(cls.tree)

    def test_full64_guard_every_position_invalid_requests_and_clamped_tail_loads(self):
        for rows in (1, 63, 64, 65, 127, 128, 129):
            t2r = (1,) * rows
            for base in range(0, rows + 64, 64):
                actual, offsets = self.replay.guard(t2r, base, 2)
                self.assertEqual(actual, independent_guard(t2r, base, 2))
                self.assertEqual(offsets, [min(base + i, rows - 1) * 4 for i in range(64)])
        for i in range(64):
            t2r = [1] * 192
            t2r[64 + i] = 0
            self.assertFalse(self.replay.guard(tuple(t2r), 64, 2)[0], i)
        for invalid in (-1, 2, 0x7FFFFFFF, -0x80000000):
            self.assertFalse(self.replay.guard((invalid,) * 64, 0, 2)[0])

    def test_actual_producer_request_base64_eight_gathers_wait0_then_barrier(self):
        pages = 128
        source = bytes((i * 37 + i // 256) % 256 for i in range(pages * 1024))
        tables = tuple(tuple((r * 17 + p * 13) % pages for p in range(33)) for r in range(4))
        t2r = sum(((r,) * 64 for r in range(4)), ())
        for rblk in range(4):
            for origin in (0, 64, 128, 192):
                records, events, loads = self.replay.stage(t2r, tables, rblk, origin, pages)
                unused, expected, writes = stage_bytes(source, tables[rblk], origin)
                self.assertEqual(records, expected)
                self.assertEqual(events, [(wave, kind, *args) for wave in range(4)
                                          for kind, args in (("wait", (0,)), ("barrier", ()))])
                self.assertEqual([offset for kind, offset in loads if kind == "request"], [rblk * 64 * 4] * 4)
                self.assertEqual(len([item for item in loads if item[0] == "table"]), 16)
        stage = self.replay.helpers["_pair_stage_tdm"]
        self.assertFalse(_calls(stage, "_ld_i32"))
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in ("lane", "kgrp", "plans", "pp") for n in ast.walk(stage)))
        direct = [ast.unparse(n.value) for n in stage.body if isinstance(n, ast.Expr)]
        self.assertEqual(direct[-2:], ["fx.rocdl.tdm_ops.tensor_wait(0)", "fx.gpu.barrier()"])

    def assert_replay(self, t2r, qpos, contexts, tables, pages=16, columns=65, tdm=True):
        actual = self.replay.replay(t2r, qpos, contexts, tables, pages, columns, tdm)
        rows = len(t2r)
        expected, visible = {}, {}
        for row in range(rows):
            req = t2r[row]
            valid_req = 0 <= req < len(tables)
            safe = req if valid_req else 0
            vis = max(min(qpos[row] + 1, contexts[safe]), 0) // 4 if valid_req else 0
            visible[row] = vis
            for col in range(columns):
                physical = tables[safe][col // 4]
                valid = valid_req and 0 <= physical < pages and col < vis
                expected[row * columns + col] = 1 + row * 65536 + physical * 4 + col % 4 if valid else -math.inf
        self.assertEqual(actual.output, expected)
        self.assertEqual(actual.counts, Counter({i: 1 for i in range(rows * columns)}))
        self.assertEqual(actual.vis, visible)
        self.assertEqual(actual.vis_counts, Counter({i: 1 for i in range(rows)}))
        self.assertEqual(actual.rst_counts, actual.vis_counts)
        for by in range((rows + 63) // 64):
            reuse = independent_guard(t2r, by * 64, len(tables))
            block_stages = [call for call in actual.stages if call[0] == by]
            self.assertEqual(len(block_stages), ((columns + 63) // 64) * 128 if reuse else 0)
            self.assertTrue(all(call[4] == by * 2 and call[6] == ("shared" if tdm else "keys") for call in block_stages))
            if not reuse:
                self.assertFalse(any(call[0] == by for call in actual.pairs))
                self.assertTrue(all(call[-1] == "keys" for call in actual.raw if call[0] == by))
        for by, chunk, wave, lane, qa, qb, col, ptr in actual.pairs:
            self.assertIn(qa, (by * 64 + wave * 16, by * 64 + wave * 16 + 8))
            self.assertEqual(qb, qa + 4)
            self.assertEqual(ptr, "shared" if tdm else "keys")
        return actual

    def test_actual_kernel_both_pairs_invisible_groups_column_tails_and_raw64(self):
        table = tuple((p * 7 + 3) % 16 for p in range(33))
        for columns in (1, 17, 63, 64, 65, 127):
            for tdm in (True, False):
                # First group invisible in alternating waves; second always
                # visible. Detect using first group's vmax or offsets twice.
                qpos = tuple(-1 if row % 16 < 8 and row // 16 % 2 == 0 else 527 for row in range(128))
                self.assert_replay((0,) * 128, qpos, (528,), (table,), columns=columns, tdm=tdm)
        invisible = self.assert_replay((0,) * 64, (-1,) * 64, (0,), (table,))
        self.assertFalse(invisible.pairs)
        self.assertEqual(len(invisible.stages), 256, "one shared stage per block/chunk even if all pairs invisible")

    def test_actual_kernel_mixed_requests_invalid_requests_pages_and_partial64(self):
        tables = tuple(tuple((3, -1, 16, 0x7FFFFFFF, 0, 15, -0x80000000)[(p + r) % 7]
                             for p in range(33)) for r in range(3))
        for rows in (1, 3, 4, 15, 16, 31, 32, 63, 64, 65, 127, 129):
            t2r = tuple((0, 1, 0, 2, -1, 3, 1, 2)[r % 8] for r in range(rows))
            qpos = tuple((-9, -1, 3, 15, 63, 263, 527)[r % 7] for r in range(rows))
            with self.subTest(rows=rows):
                self.assert_replay(t2r, qpos, (528, 264, 0), tables)
        # One reusable block followed by a one-row raw fallback.
        self.assert_replay((1,) * 65, (527,) * 65, (528, 528, 528), tables)

    def test_actual_query_helper_all_waves_four_mtiles_and_clamped_row_tail(self):
        rows = 129
        source = struct.pack("<%dH" % (rows * 4 * 128),
                            *((i * 73 + i // 128 * 11) & 0xFFFF for i in range(rows * 4 * 128)))
        ns = self.replay.namespace()
        def fragment(data, offset):
            self.assertEqual(offset % 16, 0)
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(offset + 48, len(data))
            return data[offset:offset + 16] + data[offset + 32:offset + 48]
        ns.update(_frag=fragment, q_row_bytes=1024, head_bytes=256, kstep_bytes=64)
        _exec([self.replay.helpers["_q_frags"]], ns)
        for block in (0, 1, 2):
            for wave in range(4):
                for mi in range(4):
                    t0 = block * 64 + wave * 16 + mi * 4
                    for lane in range(32):
                        actual = ns["_q_frags"](source, t0, rows, lane % 16, lane // 16)
                        row, head = min(t0 + lane % 16 // 4, rows - 1), lane % 4
                        self.assertEqual(actual, [query_reference(source, row, head, lane, step)
                                                  for step in range(4)], (block, wave, mi, lane))

    def test_production_geometry_stage_once_and_sequential_pair_source(self):
        kernel = self.replay.helpers["kernel_r64"]
        assignments = {t.id: n.value for n in kernel.body if isinstance(n, ast.Assign)
                       for t in n.targets if isinstance(t, ast.Name)}
        for rblk in (0, 1, 17):
            for wave in range(4):
                self.assertEqual(_int_expr(assignments["t0"], dict(self.replay.constants, rblk=rblk, wave=wave)), rblk * 64 + wave * 16)
        for name in ("metas", "plans", "offs"):
            self.assertIsInstance(assignments[name], ast.ListComp)
            self.assertEqual(ast.unparse(assignments[name].generators[0].iter), "range_constexpr(4)")
        reuse = next(n for n in kernel.body if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse")
        self.assertEqual(len(_calls(reuse, "pair_stage")), 1)
        self.assertEqual(ast.unparse(reuse.body[0]), "pair_stage(KV, kptr, pt_rsrc, t2r_rsrc, rblk * 2, col0, width, num_pages)")
        loop = next(n for n in reuse.body if isinstance(n, ast.For))
        self.assertEqual(ast.unparse(loop.iter), "range_constexpr(2)")
        self.assertFalse(_calls(loop, "pair_stage"))
        self.assertEqual(len(_calls(loop, "_q_frags")), 2)
        self.assertEqual(len(_calls(loop, "_pair_values")), 1)
        fallback = next(n for n in kernel.body if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse == 0")
        self.assertFalse(_calls(fallback, "pair_stage"))
        self.assertEqual(ast.unparse(fallback.body[0].iter), "range_constexpr(4)")
        self.assertEqual(ast.unparse(_calls(fallback, "_pass_values")[0].args[1]), "kv_rsrc")
        self.assertEqual(len(_calls(fallback, "_pass_values")[0].args), 10, "no staged flag in raw fallback")
        build = self.replay.functions["_build_wmma_kernel"]
        guard_load = next(n.value for n in build.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "guard_load" for t in n.targets))
        self.assertEqual(ast.unparse(guard_load), "_ld_uniform_i32 if use_tdm else _ld_i32")
        self.assertFalse(_calls(self.replay.helpers["_r64_guard"], "_ld_i32"))
        storage = self.replay.helpers["_pair_storage_tdm"]
        arrays = [n for n in ast.walk(storage) if isinstance(n, ast.Subscript) and ast.unparse(n.value) == "fx.Array"]
        self.assertEqual(len(arrays), 1)
        kind, count, alignment = arrays[0].slice.elts
        self.assertEqual(_int_expr(count, self.replay.constants) * 4, 17408)
        self.assertGreaterEqual(_int_expr(alignment, {}), 128)


class BM64HostRoutingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.replay = SourceReplay(cls.tree)

    def namespace(self, mode, access="raw", r64=False):
        def kernel(*, name, known_block_size):
            def capture(fn):
                return SimpleNamespace(_func=fn, name=name, block=known_block_size)
            return capture
        ns = self.replay.namespace()
        ns.update(_BF16_K_LOAD=mode, _ACCESS=access, _BF16_R64=r64,
                  _ld_i32=None, _ld_uniform_i32=None,
                  flyc=SimpleNamespace(kernel=kernel, jit=lambda fn: fn),
                  fx=SimpleNamespace(Stream=lambda value: value))
        names = ("_resolve_bf16_k_load", "wmma_rows_per_block", "wmma_bf16_rows_per_block",
                 "wmma_grid", "_build_wmma_kernel")
        _exec([self.replay.functions[name] for name in names], ns)
        return ns

    def test_shape_threshold64_requires_opt_in_tdm_raw_access_and_legacy_unchanged(self):
        for r64 in (False, True):
            for mode in ("auto", "raw", "tdm"):
                for access in ("raw", "cop"):
                    ns = self.namespace(mode, access, r64)
                    for requests in (1, 2, 16):
                        for rows in (1, 16, 31, 32, 63, 64, 127, 128, 512, 2880,
                                     64 * requests - 1, 64 * requests):
                            legacy = 32 if rows >= 32 and rows >= 32 * requests else 16
                            expected = (64 if r64 and mode != "raw" and access == "raw"
                                        and rows >= 64 * requests else legacy)
                            with self.subTest(r64=r64, mode=mode, access=access,
                                              rows=rows, requests=requests):
                                self.assertEqual(ns["wmma_rows_per_block"](rows, requests), legacy)
                                self.assertEqual(ns["wmma_bf16_rows_per_block"](rows, requests), expected)
                                self.assertEqual(ns["wmma_grid"](rows, 65, expected),
                                                 (2, (rows + expected - 1) // expected))

    def test_real_builder64_modes_and_copy_access_never_execute_kernel(self):
        for mode in ("auto", "raw", "tdm"):
            for access in ("raw", "cop"):
                ns = self.namespace(mode, access)
                for explicit in (None, "auto", "raw", "tdm"):
                    launch = ns["_build_wmma_kernel"](1.0, 64, explicit)
                    closure = dict(zip(launch.__code__.co_freevars,
                                       (cell.cell_contents for cell in launch.__closure__ or ())))
                    selected = closure["selected_kernel"]
                    resolved = mode if explicit is None else explicit
                    tdm = resolved != "raw" and access == "raw"
                    self.assertEqual(selected._func.__name__, "kernel_r64")
                    self.assertEqual(selected.block, [128, 1, 1])
                    self.assertEqual(launch.dtype_tag, "bf16_wmma")
                    self.assertEqual(launch.kernel_name, "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r64_c64_flydsl" + ("_tdm" if tdm else ""))
                    self.assertEqual(selected.name, launch.kernel_name)
                self.assertEqual(ns["_resolve_bf16_k_load"](64), "raw" if mode == "raw" else "tdm")
                for invalid in (0, 4, 31, 63, 128):
                    with self.assertRaises(ValueError): ns["_build_wmma_kernel"](1.0, invalid)
                    with self.assertRaises(ValueError): ns["wmma_grid"](65, 65, invalid)

    def test_wrapper64_resolution_and_unaligned_safe_actual_raw_fallback(self):
        wrapper = self.replay.functions["qsa_paged_mqa_logits_page4"]
        branch = next(n for n in wrapper.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "not is_fp8 and _BF16_PATH == 'wmma'")
        end = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "_run")
        for r64 in (False, True):
            for mode in ("auto", "raw", "tdm"):
                for access in ("raw", "cop"):
                    for rows, requests in ((128, 2), (64, 16)):
                        for pointer in (0x1000, 0x1010, 0x1070):
                            ns = self.namespace(mode, access, r64)
                            ns.update(rows=rows, num_requests=requests, columns=65, divisor=1.0,
                                      kv_u8=SimpleNamespace(data_ptr=lambda p=pointer: p))
                            original = ns["wmma_bf16_rows_per_block"](rows, requests)
                            resolved = ns["_resolve_bf16_k_load"](original)
                            fallback = resolved == "tdm" and pointer % 128 != 0
                            _exec(branch.body[:end], ns)
                            expected = (16, "raw") if fallback else (original, resolved)
                            with self.subTest(r64=r64, mode=mode, access=access,
                                              rows=rows, pointer=pointer):
                                self.assertEqual((ns["rows_per_block"], ns["k_load"]), expected)
                                cols = ns["launch"].cols_per_block
                                self.assertEqual((ns["grid_x"], ns["grid_y"]),
                                                 ((65 + cols - 1) // cols,
                                                  (rows + expected[0] - 1) // expected[0]))
        fp8 = next(n for n in wrapper.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "is_fp8 and _FP8_PATH == 'wmma'")
        self.assertFalse(_calls(fp8, "wmma_bf16_rows_per_block"))
        self.assertEqual(len(_calls(fp8, "wmma_rows_per_block")), 1)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: MIT
"""Independent, stdlib-only byte/AST checks of the fp8 r32 padded K tile.

A 64x128 E4M3 tile occupies 9216 LDS bytes: each 128-byte source row is
followed by 16 untouched bytes. Four producer waves issue two eight-i32 gathers
each. Consumers read aligned b128 at row*144+s*32 and +16, selecting the low
or high eight bytes according to lane//16, before the existing exact converter.
Invalid pages copy physical page zero, retaining c%4; masks never zero data.

The reference computes E4M3 bf16 encodings using integer field arithmetic,
independently of both the half-rescale decoder and production helper AST. The
small DSL mock executes selected source helpers, never imports aiter/FlyDSL/
torch, and never compiles/launches GPU code. Request/store tests are structural
and address/mask models, not GEMM numerical or hardware/lowering evidence.

Run directly (avoid the GPU-touching aiter package import):
    python -B aiter/ops/flydsl/kernels/mqa_logits/_g2/test_fp8_k_tdm_cpu.py -v
"""
from __future__ import annotations

import sys as _n64_sys, pathlib as _n64_path  # ROUND-NATIVE64 seam support
_n64_sys.path.insert(0, str(_n64_path.Path(__file__).resolve().parent))
from native64_seam import ensure_i32  # noqa: E402

import ast
from dataclasses import dataclass
from pathlib import Path
import random
import struct
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
PAGE_ROWS, HEAD_DIM, TILE_ROWS = 4, 128, 64
ROW_BYTES, PAGE_BYTES, LDS_PITCH, LDS_BYTES = 128, 512, 144, 9216
PAD_INTERVAL_DWORDS, PAD_AMOUNT_DWORDS = 32, 4
WAVES, LANES, KCHUNKS, GROUP_ROWS = 4, 32, 4, 8
VALUES = {
    "PAGE_SIZE": 4, "NUM_HEADS": 4, "HEAD_DIM": 128, "WMMA_WAVE": 32,
    "WMMA_WAVES": 4, "WMMA_THREADS": 128, "WMMA_ROWS_PER_WAVE": 4,
    "WMMA_PASSES": 4, "WMMA_N": 16, "WMMA_K": 32, "WMMA_K_STEPS": 4,
    "WMMA_N_TILES": 4, "WMMA_COLS": 64,
    "TDM_FP8_PAD_INTERVAL_DWORDS": 32, "TDM_FP8_PAD_AMOUNT_DWORDS": 4,
    "TDM_FP8_K_ROW_BYTES": 144, "TDM_FP8_K_TILE_BYTES": 9216,
}


def reference_bf16(code):
    """Integer E4M3 field reference, including the kernel's finite +/-480 codes."""
    sign = (code & 128) << 8
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent:
        return sign | ((exponent - 7 + 127) << 7) | (mantissa << 4)
    if not mantissa:
        return sign
    significant = mantissa.bit_length()
    return (sign | ((significant - 10 + 127) << 7)
            | ((mantissa << (8 - significant)) - 128))


def half_decode_word(word):
    """Independent packed-half route used by the CPU layout model."""
    out = []
    for position in range(4):
        code = (word >> (8 * position)) & 255
        half_bits = ((code & 128) << 8) | ((code & 127) << 7)
        value = struct.unpack("<e", struct.pack("<H", half_bits))[0] * 256.0
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert bits & 65535 == 0
        out.append(bits >> 16)
    return struct.pack("<4H", *out)


def convert_fragment(payload):
    assert len(payload) == 16
    return b"".join(half_decode_word(word) for word in struct.unpack("<4I", payload))


@dataclass(frozen=True)
class RowMask:
    logical_ok: bool
    page_ok: bool
    col_ok: bool
    keep: bool


@dataclass(frozen=True)
class Issue:
    wave: int
    group: int
    local_row: int
    lds_byte_offset: int
    row_ids: tuple[int, ...]


def stage_tile(source, table, origin=0, columns=None, visible=None, req_ok=True,
               *, issue_order=None, scratch=None):
    """Descriptor stream model: copy DWORDs, skipping four after every 32."""
    assert source and len(source) % PAGE_BYTES == 0
    assert table and origin >= 0 and origin % TILE_ROWS == 0
    columns = len(table) * PAGE_ROWS if columns is None else columns
    visible = columns if visible is None else visible
    assert 0 < columns <= len(table) * PAGE_ROWS
    num_pages = len(source) // PAGE_BYTES
    order = tuple(range(8)) if issue_order is None else tuple(issue_order)
    assert sorted(order) == list(range(8))
    tile = bytearray(bytes([0xA5]) * LDS_BYTES if scratch is None else scratch)
    assert len(tile) == LDS_BYTES
    issues, masks, writes = [], {}, bytearray(LDS_BYTES)
    for identifier in order:
        wave, group = divmod(identifier, 2)
        local = wave * 16 + group * 8
        row_ids = []
        # Match descriptor planning granularity: two page loads, four row IDs each.
        for page in range(2):
            logical = (origin + local) // PAGE_ROWS + page
            logical_ok = logical < len(table)
            physical = table[logical if logical_ok else 0]
            valid = req_ok and logical_ok and 0 <= physical < num_pages
            physical = physical if valid else 0
            for off in range(PAGE_ROWS):
                c = origin + local + page * PAGE_ROWS + off
                row_ids.append(physical * PAGE_ROWS + off)
                masks[c - origin] = RowMask(logical_ok, valid, c < columns,
                                            valid and c < visible)
        row_ids = struct.unpack("<8i", struct.pack("<8i", *row_ids))
        start = local * LDS_PITCH
        issues.append(Issue(wave, group, local, start, row_ids))
        for data_dw in range(8 * PAD_INTERVAL_DWORDS):
            row, dw_in_row = divmod(data_dw, PAD_INTERVAL_DWORDS)
            src = row_ids[row] * ROW_BYTES + dw_in_row * 4
            dst = start + (data_dw + (data_dw // PAD_INTERVAL_DWORDS)
                           * PAD_AMOUNT_DWORDS) * 4
            assert 0 <= src <= len(source) - 4
            assert start <= dst <= start + 8 * LDS_PITCH - 4
            assert 0 <= dst <= len(tile) - 4
            tile[dst:dst + 4] = source[src:src + 4]
            for byte in range(dst, dst + 4):
                writes[byte] += 1
    return bytes(tile), tuple(issues), tuple(masks[i] for i in range(64)), writes


def raw_fragment(source, table, origin, nt, chunk, lane, columns, visible, req_ok=True):
    """Independent global-load fallback; no producer/descriptor plan is reused."""
    c = origin + nt * 16 + lane % 16
    logical_ok = c // 4 < len(table)
    raw_page = table[c // 4 if logical_ok else 0]
    page_ok = bool(req_ok and logical_ok and 0 <= raw_page < len(source) // 512)
    page = raw_page if page_ok else 0
    base = page * 512 + (c % 4) * 128 + (lane // 16) * 8 + chunk * 32
    assert 0 <= base and base + 24 <= len(source)
    payload = source[base:base + 8] + source[base + 16:base + 24]
    bf16 = struct.pack("<16H", *(reference_bf16(code) for code in payload))
    return payload, bf16, RowMask(logical_ok, page_ok, c < columns, page_ok and c < visible)


def lds_fragment(tile, nt, chunk, lane):
    base = (nt * 16 + lane % 16) * LDS_PITCH + chunk * 32
    lo, hi = base, base + 16
    assert lo % 16 == hi % 16 == 0
    assert 0 <= lo <= len(tile) - 16 and 0 <= hi <= len(tile) - 16
    pick = (lane // 16) * 8
    payload = tile[lo + pick:lo + pick + 8] + tile[hi + pick:hi + pick + 8]
    return payload, convert_fragment(payload), (lo, hi)


def full_single_request(t2r, base, num_requests):
    return (base + 32 <= len(t2r) and 0 <= t2r[base] < num_requests
            and all(req == t2r[base] for req in t2r[base:base + 32]))


def calls(root, name):
    return [node for node in ast.walk(root) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def assignments(root, *, recursive=True):
    nodes = ast.walk(root) if recursive else root.body
    return {target.id: node.value for node in nodes if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def integer_expression(node, values):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp):
        a, b = integer_expression(node.left, values), integer_expression(node.right, values)
        operations = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b,
                      ast.Mult: lambda: a * b, ast.FloorDiv: lambda: a // b,
                      ast.Mod: lambda: a % b}
        if type(node.op) in operations:
            return operations[type(node.op)]()
    raise AssertionError("unsupported expression: " + ast.unparse(node))


class Bool:
    def __init__(self, value):
        self.value = bool(value)

    def select(self, yes, no):
        return yes if self.value else no

    def __bool__(self):
        return self.value

    def __and__(self, other):
        return Bool(self.value and bool(other))

    __rand__ = __and__

    def __or__(self, other):
        return Bool(self.value or bool(other))


class I32(int):
    """Small scalar DSL value, with signed wrap for packed converter bit ops."""
    def __new__(cls, value):
        value = int(value) & 0xFFFFFFFF
        return int.__new__(cls, value - 0x100000000 if value >= 0x80000000 else value)

    def ir_value(self):
        return self

    __add__ = lambda s, o: I32(int(s) + int(o))
    __radd__ = __add__
    __sub__ = lambda s, o: I32(int(s) - int(o))
    __rsub__ = lambda s, o: I32(int(o) - int(s))
    __mul__ = lambda s, o: I32(int(s) * int(o))
    __rmul__ = __mul__
    __floordiv__ = lambda s, o: I32(int(s) // int(o))
    __mod__ = lambda s, o: I32(int(s) % int(o))
    __lshift__ = lambda s, o: I32(int(s) << int(o))
    __rshift__ = lambda s, o: I32(int(s) >> int(o))
    __and__ = lambda s, o: I32(int(s) & int(o))
    __rand__ = __and__
    __or__ = lambda s, o: I32(int(s) | int(o))
    __ror__ = __or__
    __eq__ = lambda s, o: Bool(int(s) == int(o))
    __ne__ = lambda s, o: Bool(int(s) != int(o))
    __lt__ = lambda s, o: Bool(int(s) < int(o))
    __le__ = lambda s, o: Bool(int(s) <= int(o))
    __ge__ = lambda s, o: Bool(int(s) >= int(o))
    __gt__ = lambda s, o: Bool(int(s) > int(o))
    __hash__ = int.__hash__


class Vector:
    def __init__(self, values, dtype="i32"):
        if isinstance(values, Vector):
            self.values, self.dtype = values.values, values.dtype
        else:
            self.values, self.dtype = list(values), dtype

    def __getitem__(self, index):
        return self.values[int(index)]

    @classmethod
    def from_elements(cls, elements, dtype):
        return cls(elements, dtype)

    @classmethod
    def filled(cls, count, value, dtype):
        return cls([value] * count, dtype)

    @staticmethod
    def make_type(count, dtype):
        return count, dtype

    def bitcast(self, dtype):
        if self.dtype is I32 and dtype == "f16":
            bits = b"".join(struct.pack("<I", int(v) & 0xFFFFFFFF) for v in self.values)
            return Vector(struct.unpack("<" + "e" * (len(bits) // 2), bits), "f16")
        if self.dtype == "f32" and dtype is I32:
            return Vector([I32(struct.unpack("<I", struct.pack("<f", v))[0])
                           for v in self.values], I32)
        raise AssertionError((self.dtype, dtype))

    def to(self, dtype):
        assert self.dtype == "f16" and dtype == "f32"
        return Vector([float(v) for v in self.values], dtype)

    def __mul__(self, other):
        assert self.dtype == other.dtype == "f16"
        # Real half multiplication, including signed zeros and preserved denormals.
        return Vector([struct.unpack("<e", struct.pack("<e", a * b))[0]
                       for a, b in zip(self.values, other.values)], "f16")


def exec_helpers(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ensure_i32(namespace)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class FP8ByteTileCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = list(range(256)) * 256
        random.Random(0xE4_144).shuffle(data)
        cls.source = bytes(data)
        cls.num_pages = len(cls.source) // PAGE_BYTES

    def assert_equivalent(self, table, origin=0, columns=None, visible=None,
                          req_ok=True, **kwargs):
        columns = len(table) * 4 if columns is None else columns
        visible = columns if visible is None else visible
        tile, issues, masks, writes = stage_tile(self.source, table, origin, columns,
                                                visible, req_ok, **kwargs)
        owners, restored = bytearray(8192), bytearray(8192)
        for nt in range(4):
            for chunk in range(4):
                for lane in range(32):
                    reference, bf16, mask = raw_fragment(self.source, table, origin, nt,
                                                         chunk, lane, columns, visible, req_ok)
                    payload, decoded, offsets = lds_fragment(tile, nt, chunk, lane)
                    self.assertEqual(payload, reference, (nt, chunk, lane))
                    self.assertEqual(decoded, bf16, (nt, chunk, lane))
                    row = nt * 16 + lane % 16
                    self.assertEqual(masks[row], mask)
                    self.assertEqual(offsets, (row * 144 + chunk * 32,
                                               row * 144 + chunk * 32 + 16))
                    for v, code in enumerate(reference):
                        k = chunk * 32 + (v // 8) * 16 + (lane // 16) * 8 + v % 8
                        element = row * 128 + k
                        owners[element] += 1
                        restored[element] = code
        self.assertEqual(owners, bytearray([1]) * 8192)
        self.assertEqual(bytes(restored), b"".join(tile[row * 144:row * 144 + 128]
                                                for row in range(64)))
        canary = kwargs.get("scratch", bytes([0xA5]) * LDS_BYTES)
        for row in range(64):
            self.assertEqual(writes[row * 144:row * 144 + 128], bytearray([1]) * 128)
            self.assertEqual(writes[row * 144 + 128:(row + 1) * 144], bytearray(16))
            self.assertEqual(tile[row * 144 + 128:(row + 1) * 144],
                             canary[row * 144 + 128:(row + 1) * 144])
        return tile, issues, masks

    def test_all_256_codes_in_all_four_dword_positions(self):
        for code in range(256):
            for position in range(4):
                payload = bytearray([0x00, 0x80, 0x7F, 0xFF])
                payload[position] = code
                actual = half_decode_word(int.from_bytes(payload, "little"))
                self.assertEqual(actual, struct.pack("<4H", *(reference_bf16(b) for b in payload)),
                                 (code, position))
        self.assertEqual([reference_bf16(c) for c in (0, 128, 1, 129, 127, 255)],
                         [0, 0x8000, 0x3B00, 0xBB00, 0x43F0, 0xC3F0])

    def test_all_lane_kchunk_wave_fragments_and_complete_code_sweep(self):
        table = tuple(range(self.num_pages))
        restored = []
        for origin in range(0, self.num_pages * 4, 64):
            tile, issues, _ = self.assert_equivalent(table, origin)
            restored.append(b"".join(tile[row * 144:row * 144 + 128] for row in range(64)))
            self.assertEqual([(i.wave, i.group, i.local_row, i.lds_byte_offset) for i in issues],
                             [(w, g, w * 16 + g * 8, (w * 16 + g * 8) * 144)
                              for w in range(4) for g in range(2)])
        self.assertEqual(b"".join(restored), self.source)
        self.assertEqual(set(b"".join(restored)), set(range(256)))

    def test_invalid_physical_pages_clamp_to_zero_and_retain_c_mod_four(self):
        table = (7, -1, self.num_pages, self.num_pages + 17,
                 0x7FFFFFFF, -0x80000000, 0, self.num_pages - 1) * 4
        for origin in (0, 64):
            tile, issues, masks = self.assert_equivalent(table, origin)
            ids = sum((issue.row_ids for issue in issues), ())
            for row in range(64):
                c = origin + row
                valid = 0 <= table[c // 4] < self.num_pages
                self.assertEqual(masks[row].page_ok, valid)
                if not valid:
                    self.assertEqual(ids[row], c % 4)
                    self.assertEqual(tile[row * 144:row * 144 + 128],
                                     self.source[(c % 4) * 128:(c % 4 + 1) * 128])

    def test_column_and_logical_page_tails(self):
        for width in (1, 3, 15, 16, 17, 19, 31, 33, 65):
            table = tuple((i * 29 + 7) % self.num_pages for i in range(width))
            for columns in sorted({1, max(1, width * 4 - 3), width * 4}):
                origins = {0, ((columns - 1) // 64) * 64, ((width * 4 + 63) // 64) * 64}
                for origin in sorted(origins):
                    with self.subTest(width=width, columns=columns, origin=origin):
                        _, issues, masks = self.assert_equivalent(table, origin, columns)
                        ids = sum((issue.row_ids for issue in issues), ())
                        for row in range(64):
                            c = origin + row
                            self.assertEqual(masks[row].col_ok, c < columns)
                            if c // 4 >= width:
                                self.assertFalse(masks[row].page_ok)
                                self.assertEqual(ids[row], c % 4)

    def test_visibility_request_masks_never_zero_or_suppress_staged_payloads(self):
        table = (7, -1, self.num_pages, 3) * 4
        baseline = self.assert_equivalent(table, columns=61, visible=64)[0]
        for visible in (0, 1, 4, 17, 60, 61, 64):
            tile, _, masks = self.assert_equivalent(table, columns=61, visible=visible)
            self.assertEqual(tile, baseline)
            self.assertEqual(tuple(m.keep for m in masks),
                             tuple(m.page_ok and c < visible for c, m in enumerate(masks)))
        _, issues, masks = self.assert_equivalent(table, req_ok=False)
        self.assertEqual(sum((i.row_ids for i in issues), ()), tuple(c % 4 for c in range(64)))
        self.assertFalse(any(m.page_ok or m.keep for m in masks))

    def test_issue_permutations_preserve_all_padding_and_do_not_overlap(self):
        rng = random.Random(0x9216)
        scratch = bytes(rng.randrange(256) for _ in range(LDS_BYTES))
        table = tuple((i * 53 + 11) % self.num_pages for i in range(32))
        orders = [tuple(range(8)), tuple(reversed(range(8))), (0, 2, 4, 6, 1, 3, 5, 7)]
        for _ in range(5):
            order = list(range(8))
            rng.shuffle(order)
            orders.append(tuple(order))
        expected = None
        for order in orders:
            tile, issues, _ = self.assert_equivalent(table, origin=64, scratch=scratch,
                                                    issue_order=order)
            expected = tile if expected is None else expected
            self.assertEqual(tile, expected)
            ranges = sorted((i.lds_byte_offset, i.lds_byte_offset + 8 * 144) for i in issues)
            self.assertEqual(ranges, [(i * 1152, (i + 1) * 1152) for i in range(8)])
            self.assertTrue(all(i.lds_byte_offset % 128 == 0 for i in issues))
            self.assertEqual(ranges[-1][-1], 9216)


class FP8KernelSourceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.top = {n.name: n for n in cls.tree.body if isinstance(n, ast.FunctionDef)}
        cls.build = cls.top["_build_wmma_fp8_kernel"]
        cls.helpers = {n.name: n for n in cls.build.body if isinstance(n, ast.FunctionDef)}
        cls.fx = SimpleNamespace(Int32=I32, Float16="f16", Float32="f32")
        cls.ns = dict(VALUES, fx=cls.fx, Vec=Vector, range_constexpr=range,
                      _E4M3_F16_MAG=I32(0x3F803F80), _E4M3_F16_SIGN=I32(0x80008000),
                      _BF16_LO_MASK=I32(0xFFFF), _BF16_HI_MASK=I32(0xFFFF0000),
                      _imin=lambda a, b: I32(min(int(a), int(b))),
                      _imax=lambda a, b: I32(max(int(a), int(b))),
                      guard_load=lambda plane, offset: I32(plane[int(offset) // 4]))
        exec_helpers([cls.top["_e4m3x4_to_bf16x2"]], cls.ns)

    def test_actual_converter_all_codes_every_byte_position(self):
        for code in range(256):
            for position in range(4):
                payload = bytearray([0, 128, 127, 255])
                payload[position] = code
                lo, hi = self.ns["_e4m3x4_to_bf16x2"](I32(int.from_bytes(payload, "little")))
                actual = struct.pack("<2I", int(lo) & 0xFFFFFFFF, int(hi) & 0xFFFFFFFF)
                self.assertEqual(actual, struct.pack("<4H", *(reference_bf16(b) for b in payload)),
                                 (code, position))

    def test_layout_constants_and_descriptor_element_counts(self):
        top_values = assignments(self.tree)
        for name in ("TDM_FP8_PAD_INTERVAL_DWORDS", "TDM_FP8_PAD_AMOUNT_DWORDS",
                     "TDM_FP8_K_ROW_BYTES", "TDM_FP8_K_TILE_BYTES"):
            self.assertEqual(integer_expression(top_values[name], VALUES), VALUES[name], name)
        gather = self.top["_tdm_fp8_k_gather"]
        descriptors = calls(gather, "tdm_ops.make_tensor_gather_descriptor")
        self.assertEqual(len(descriptors), 1)
        kwargs = {k.arg: k.value for k in descriptors[0].keywords}
        expected = {"row_width": 128, "tensor_dim0": 128, "stride": 128,
                    "elem_bytes": 1, "index_size": 32, "pad_interval": 128,
                    "pad_amount": 16, "workgroup_mask": 0}
        for name, value in expected.items():
            self.assertEqual(integer_expression(kwargs[name], VALUES), value, name)
        self.assertEqual(ast.unparse(kwargs["lds_byte_offset"]), "fx.Index(byte_off)")
        # Stock converts element counts to DWORDs before encoding: log2(I)-1,
        # A-1. The separate source/compile gate verifies stock dev886 lowering.
        interval_dw = expected["pad_interval"] * expected["elem_bytes"] // 4
        amount_dw = expected["pad_amount"] * expected["elem_bytes"] // 4
        self.assertEqual((interval_dw, amount_dw), (32, 4))
        self.assertEqual((interval_dw.bit_length() - 2, amount_dw - 1), (4, 3))
        self.assertEqual(len(calls(gather, "tdm_ops.tensor_load_gather")), 1)
        bf16 = {k.arg: k.value for k in calls(self.top["_tdm_k_gather"],
                                             "tdm_ops.make_tensor_gather_descriptor")[0].keywords}
        bf16_values = dict(VALUES, TDM_PAD_INTERVAL_DWORDS=64, TDM_PAD_AMOUNT_DWORDS=4)
        self.assertEqual([integer_expression(bf16[k], bf16_values)
                          for k in ("elem_bytes", "pad_interval", "pad_amount")], [2, 128, 8])

    def test_allocator_is_exactly_9216_bytes(self):
        storage = self.helpers["_pair_storage_tdm"]
        arrays = [n for n in ast.walk(storage) if isinstance(n, ast.Subscript)
                  and ast.unparse(n.value) == "fx.Array"]
        self.assertEqual(len(arrays), 1)
        dtype, count, alignment = arrays[0].slice.elts
        self.assertEqual(ast.unparse(dtype), "fx.Int32")
        self.assertEqual(integer_expression(count, VALUES) * 4, LDS_BYTES)
        self.assertGreaterEqual(integer_expression(alignment, VALUES), 128)
        allocators = calls(storage, "fx.SharedAllocator")
        self.assertEqual(len(allocators), 1)
        kwargs = {k.arg: k.value for k in allocators[0].keywords}
        self.assertIs(kwargs["static"].value, True)
        self.assertGreaterEqual(integer_expression(kwargs["base_alignment"], VALUES), 128)

    def test_actual_producer_rows_offsets_clamps_and_wave_issue_permutations(self):
        stage = self.helpers["_pair_stage_tdm"]
        bound = assignments(stage)
        self.assertEqual(ast.unparse(bound["wave_id"]), "fx.Int32(fx.rocdl.wave_id())")
        self.assertFalse(calls(stage, "_ld_i32"))
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in ("lane", "kgrp", "pp", "plans")
                             for n in ast.walk(stage)))
        loops = {n.target.id: n for n in ast.walk(stage) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name)}
        self.assertEqual(ast.unparse(loops["group"].iter), "range_constexpr(2)")
        self.assertEqual(ast.unparse(loops["page"].iter), "range_constexpr(2)")
        ns = dict(self.ns)
        events, loads = [], []
        wave_box = [0]
        fx = SimpleNamespace(Int32=I32,
                             rocdl=SimpleNamespace(wave_id=lambda: wave_box[0],
                                  tdm_ops=SimpleNamespace(tensor_wait=lambda x: events.append(("wait", int(x))))),
                             gpu=SimpleNamespace(barrier=lambda: events.append(("barrier",))))
        def load(plane, offset):
            index = int(offset) // 4
            self.assertEqual(int(offset) % 4, 0)
            self.assertTrue(0 <= index < len(plane))
            loads.append(index)
            return I32(plane[index])
        ns.update(fx=fx, _ld_uniform_i32=load,
                  _tdm_fp8_k_gather=lambda kv, ptr, ids, pages, off:
                      events.append(("gather", tuple(map(int, ids)), int(off))))
        exec_helpers([stage], ns)
        table = (7, -1, 16, 0x7FFFFFFF, -0x80000000, 3, 0, 15) * 3
        t2r = (1,) * 64
        flat = (11,) * len(table) + table
        source = bytes(range(256)) * 32
        for origin in (0, 64, 128):
            expected = stage_tile(source, table, origin, columns=93)[1]
            for order in ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2)):
                events.clear()
                for wave in order:
                    wave_box[0] = wave
                    ns["_pair_stage_tdm"](None, None, flat, t2r, I32(1), I32(origin),
                                           I32(len(table)), I32(16))
                actual = sorted((event[2], event[1]) for event in events if event[0] == "gather")
                self.assertEqual(actual, [(i.lds_byte_offset, i.row_ids) for i in expected])
                self.assertEqual(events.count(("wait", 0)), 4)
                self.assertEqual(events.count(("barrier",)), 4)
        self.assertTrue(loads)

    def test_actual_b128_consumer_each_wave_lane_chunk_and_padding_exclusion(self):
        source = bytes(range(256)) * 64
        table = tuple((i * 13 + 7) % 32 for i in range(16))
        tile = stage_tile(source, table)[0]
        ns, addresses = dict(self.ns), []
        def ptr_load(offset, result_type):
            base = int(offset) * 4
            self.assertEqual(base % 16, 0)
            self.assertTrue(0 <= base <= LDS_BYTES - 16)
            self.assertEqual(result_type, (4, I32))
            self.assertLessEqual(base % 144 + 16, 128, "load must never touch padding")
            addresses.append(base)
            return Vector([I32(v) for v in struct.unpack("<4I", tile[base:base + 16])], I32)
        ns["fx"] = SimpleNamespace(Int32=I32, ptr_load=ptr_load)
        exec_helpers([self.helpers["_pair_base_tdm"], self.helpers["_pair_frag_tdm"]], ns)
        for wave in range(4):
            for nt in range(4):
                for chunk in range(4):
                    for lane in range(32):
                        c = 64 + nt * 16 + lane % 16
                        base = ns["_pair_base_tdm"](I32(31), I32(3), I32(c), I32(lane // 16))
                        self.assertEqual(int(base), (c % 64) * 144)
                        before = len(addresses)
                        vec = ns["_pair_frag_tdm"](I32(0), base + chunk * 32, I32(lane // 16))
                        actual = b"".join(struct.pack("<I", int(word) & 0xFFFFFFFF)
                                          for word in vec.values)
                        _, expected, _ = raw_fragment(source, table, 0, nt, chunk, lane, 64, 64)
                        self.assertEqual(actual, expected, (wave, nt, chunk, lane))
                        self.assertEqual(addresses[before:], [int(base) + chunk * 32,
                                                              int(base) + chunk * 32 + 16])

    def test_actual_raw_twins_keep_original_global_bytes_and_lane_mapping(self):
        source = bytes(range(256)) * 64
        table = tuple((i * 13 + 7) % 32 for i in range(16))
        ns = dict(self.ns)
        addresses = []
        def load(plane, byte_off, count, dtype):
            base, width = int(byte_off), int(count) * 4
            self.assertIs(dtype, I32)
            self.assertEqual(count, 2)
            self.assertTrue(0 <= base <= len(plane) - width)
            self.assertEqual(base % 8, 0)
            addresses.append(base)
            return Vector([I32(word) for word in struct.unpack("<2I", plane[base:base + width])], I32)
        ns["_ld_vec"] = load
        exec_helpers([self.helpers[n] for n in ("_kfrag_fp8", "_pair_base_raw", "_pair_frag_raw")], ns)
        for nt in range(4):
            for chunk in range(4):
                for lane in range(32):
                    c, kgrp = nt * 16 + lane % 16, lane // 16
                    phys, off = table[c // 4], c % 4
                    ns.update(page_bytes=512, key_row_bytes=128)
                    base = ns["_pair_base_raw"](I32(phys), I32(off), I32(c), I32(kgrp))
                    self.assertEqual(int(base), phys * 512 + off * 128 + kgrp * 8)
                    before = len(addresses)
                    vector = ns["_pair_frag_raw"](source, base + chunk * 32, I32(kgrp))
                    actual = b"".join(struct.pack("<I", int(word) & 0xFFFFFFFF)
                                      for word in vector.values)
                    _, expected, _ = raw_fragment(source, table, 0, nt, chunk, lane, 64, 64)
                    self.assertEqual(actual, expected, (nt, chunk, lane))
                    self.assertEqual(addresses[before:], [int(base) + chunk * 32,
                                                          int(base) + chunk * 32 + 16])

    def test_actual_wrapper_alignment_resolves_to_r16_raw_not_r16_tdm(self):
        wrapper = self.top["qsa_paged_mqa_logits_page4"]
        branch = next(n for n in wrapper.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "is_fp8 and _FP8_PATH == 'wmma'")
        stop = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "_run")
        builds = calls(branch, "_build_wmma_fp8_kernel")
        self.assertEqual(len(builds), 1)
        self.assertEqual([ast.unparse(n) for n in builds[0].args],
                         ["float(divisor)", "rows_per_block", "k_load"])
        ns = dict(self.ns)
        exec_helpers([self.top[n] for n in ("_resolve_fp8_k_load", "wmma_rows_per_block", "wmma_grid")], ns)
        for mode in ("auto", "raw", "tdm"):
            ns["_FP8_K_LOAD"] = mode
            for rows, requests in ((512, 1), (64, 16), (31, 1), (32, 1), (64, 2)):
                original = ns["wmma_rows_per_block"](rows, requests)
                resolved = "tdm" if original == 32 and mode != "raw" else "raw"
                # The public fp8 API allows 8-byte bases; TDM needs 128-byte bases.
                for pointer in (0x1000, 0x1008, 0x1010, 0x1070, 0x1080):
                    local = dict(ns, _FP8_K_LOAD=mode, rows=rows, num_requests=requests,
                                 columns=65, kv_u8=SimpleNamespace(data_ptr=lambda: pointer))
                    exec_helpers(branch.body[:stop], local)
                    expected = (16, "raw") if resolved == "tdm" and pointer % 128 else (original, resolved)
                    self.assertEqual((local["rows_per_block"], local["k_load"]), expected,
                                     (mode, rows, requests, pointer))
                    self.assertEqual((local["grid_x"], local["grid_y"]),
                                     (2, (rows + expected[0] - 1) // expected[0]))

    def test_real_full32_single_request_guard_and_tail_safe_metadata_addresses(self):
        ns = dict(self.ns)
        exec_helpers([self.helpers["_r32_guard"]], ns)
        for rows in (1, 7, 16, 31, 32, 33, 63, 64, 65, 95, 96):
            for pattern in (lambda r: 0, lambda r: 1, lambda r: -1,
                            lambda r: 4, lambda r: r % 4, lambda r: (r // 32) % 4):
                t2r = tuple(pattern(row) for row in range(rows))
                for base in range(0, rows, 32):
                    actual = ns["_r32_guard"](t2r, I32(base), I32(rows), I32(4))
                    self.assertEqual(bool(actual), full_single_request(t2r, base, 4), (rows, base))
        for changed in range(32):
            t2r = [1] * 32
            t2r[changed] = 2
            self.assertFalse(bool(ns["_r32_guard"](t2r, I32(0), I32(32), I32(4))))

    def test_tdm_stage_is_outside_per_wave_visibility_and_mixed_tail_stays_raw(self):
        kernel = self.helpers["kernel_r32"]
        reuse = next(n for n in kernel.body if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse")
        partial = next(n for n in kernel.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == "reuse == 0")
        self.assertEqual(len(calls(reuse, "pair_stage")), 1)
        stage = next(i for i, n in enumerate(reuse.body) if calls(n, "pair_stage"))
        per_wave = [i for i, n in enumerate(reuse.body) if isinstance(n, ast.If)
                    and "vmax" in ast.unparse(n.test)]
        self.assertTrue(per_wave)
        self.assertTrue(all(stage < i for i in per_wave))
        self.assertFalse(calls(partial, "pair_stage"))
        self.assertFalse(calls(partial, "fx.gpu.barrier"))
        fallback = calls(partial, "_pass_values")
        self.assertEqual(len(fallback), 1)
        self.assertEqual(ast.unparse(fallback[0].args[1]), "kv_rsrc")
        raw_r16 = self.helpers["kernel"]
        self.assertFalse(calls(raw_r16, "pair_stage"))
        stage_fn = self.helpers["_pair_stage_tdm"]
        direct = [ast.unparse(n.value.func) for n in stage_fn.body if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call)]
        self.assertEqual(direct[-2:], ["fx.rocdl.tdm_ops.tensor_wait", "fx.gpu.barrier"])
        self.assertEqual(ast.unparse(calls(stage_fn, "fx.rocdl.tdm_ops.tensor_wait")[0]),
                         "fx.rocdl.tdm_ops.tensor_wait(0)")
        pair = calls(reuse, "_pair_values")
        self.assertEqual(len(pair), 1)
        self.assertNotEqual(ast.unparse(pair[0].args[1]), "kv_rsrc")
        helper = self.helpers["_tile_pair"]
        self.assertEqual(len(calls(helper, "pair_frag")), 1)
        self.assertEqual(len(calls(helper, "fx.gemm")), 2)
        self.assertEqual(ast.unparse(calls(helper, "pair_frag")[0].args[-1]), "kgrp")

    def test_fp8_route_r16_always_raw_and_bf16_resolver_is_not_reused(self):
        ns = dict(self.ns, _FP8_K_LOAD="auto")
        exec_helpers([self.top["_resolve_fp8_k_load"]], ns)
        for environment in ("auto", "raw", "tdm"):
            ns["_FP8_K_LOAD"] = environment
            for explicit in (None, "auto", "raw", "tdm"):
                mode = environment if explicit is None else explicit
                for geometry in (16, 32):
                    self.assertEqual(ns["_resolve_fp8_k_load"](geometry, explicit),
                                     "tdm" if geometry == 32 and mode != "raw" else "raw")
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in
                             ("_resolve_bf16_k_load", "_BF16_K_LOAD") for n in ast.walk(self.build)))
        self.assertFalse(calls(self.top["_build_wmma_kernel"], "_tdm_fp8_k_gather"))
        bound = assignments(self.build, recursive=False)
        self.assertEqual(ast.unparse(bound["guard_load"]), "_ld_uniform_i32 if use_tdm else _ld_i32")
        self.assertEqual(ast.unparse(bound["pair_base"]), "_pair_base_tdm if use_tdm else _pair_base_raw")
        self.assertEqual(ast.unparse(bound["pair_frag"]), "_pair_frag_tdm if use_tdm else _pair_frag_raw")
        self.assertEqual(ast.unparse(bound["fast_cvt"]), "use_tdm and _FP8_CVT == 'fast'")
        override = next(n for n in self.build.body if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "fast_cvt")
        self.assertEqual(ast.unparse(override.body[0]), "pair_frag = _pair_frag_tdm_fast")
        # Mixed/tail/r16 helpers and bf16 never reference the conversion switch.
        for root in (self.helpers["_pair_frag_raw"], self.helpers["_kfrag_fp8"],
                     self.helpers["kernel"], self.top["_build_wmma_kernel"]):
            self.assertFalse(any(isinstance(n, ast.Name) and n.id == "_FP8_CVT"
                                 for n in ast.walk(root)))


class FP8RequestStoresCPU(unittest.TestCase):
    """Execute real request/mask/store-offset helpers, stub only GEMM scores."""
    @classmethod
    def setUpClass(cls):
        FP8KernelSourceCPU.setUpClass()
        cls.helpers = FP8KernelSourceCPU.helpers
        cls.ns = dict(FP8KernelSourceCPU.ns)
        cls.ns.update(_NEG_INF=float("-inf"), inv_divisor=1.0,
                      _RowMeta=lambda *args: SimpleNamespace(**dict(zip(
                          ("row", "row_ok", "req", "req_ok", "safe_req", "vis"), args))),
                      _Pass=lambda *args: SimpleNamespace(**dict(zip(
                          ("active", "vmax", "safe_req", "req_ok", "own_a", "own_b",
                           "vis_a", "vis_b", "kgrp"), args))),
                      _ld_i32=lambda plane, offset: I32(plane[int(offset) // 4]))
        cls.ns["fx"] = SimpleNamespace(Int32=I32, Float32=float)
        names = ("_i01", "_rows_meta", "_pass_plan", "_pass_masks", "_pair_values",
                 "_pass_values", "_store_offsets", "_lane_cols")
        exec_helpers([cls.helpers[n] for n in names], cls.ns)

    def assert_stores(self, t2r, visible, tables, columns, force_raw=False):
        rows, width, requests = len(t2r), len(tables[0]), len(tables)
        self.assertTrue(all(len(table) == width for table in tables))
        logits, counts = {}, [0] * (rows * columns)
        vis_counts, rst_counts = [0] * rows, [0] * rows
        ns = dict(self.ns)
        def page_ok(pp, c):
            logical = int(c) // 4
            return Bool(bool(pp.req_ok) and logical < width
                        and 0 <= tables[int(pp.safe_req)][logical if logical < width else 0] < 16)
        ns["_tile"] = lambda atom, kv, ks, pt, qf, pp, c, kgrp, width, pages: (
            page_ok(pp, c), float(100 + int(c)), float(200 + int(c)))
        ns["_tile_pair"] = lambda atom, kv, ks, pt, qfa, qfb, pp, c, kgrp, width, pages: (
            page_ok(pp, c), [float(100 + int(c)), float(200 + int(c))] * 2)
        # Re-exec mask helpers into this namespace so their global stubs are local.
        exec_helpers([self.helpers[n] for n in ("_pair_values", "_pass_values")], ns)
        for block in range((rows + 31) // 32):
            base = block * 32
            reuse = not force_raw and full_single_request(t2r, base, requests)
            for wave in range(4):
                t0 = base + wave * 8
                for chunk in range((columns + 63) // 64):
                    for lane in range(32):
                        group = lane // 16
                        cols = ns["_lane_cols"](I32(chunk * 64), I32(lane % 16))
                        metas = [ns["_rows_meta"](t2r, tuple(v * 4 - 1 for v in visible),
                                                 (10**6,) * requests, I32(t0 + mi * 4),
                                                 I32(rows), I32(requests)) for mi in range(2)]
                        plans = [ns["_pass_plan"](meta, I32(rows), I32(group)) for meta in metas]
                        offsets = [ns["_store_offsets"](I32(t0 + mi * 4), I32(group), cols,
                                                       I32(rows), I32(columns)) for mi in range(2)]
                        if chunk == 0 and lane < 4:
                            for mi in range(2):
                                row = t0 + mi * 4 + lane
                                if row < rows:
                                    vis_counts[row] += 1
                                    rst_counts[row] += 1
                                    self.assertEqual(int(metas[mi][lane].vis),
                                                     visible[row] if 0 <= t2r[row] < requests else 0)
                        def store(results):
                            for ok, offset, value in results:
                                index = int(offset) // 4
                                self.assertEqual(int(offset) % 4, 0)
                                self.assertTrue(0 <= index < len(counts), "even masked offset is in plane")
                                if bool(ok):
                                    counts[index] += 1
                                    # Only mask classification matters, not stubbed scores.
                                    logits[index] = value != float("-inf")
                        if reuse:
                            pa, pb = plans[0][0], plans[1][0]
                            if chunk * 64 < max(int(pa.vmax), int(pb.vmax)):
                                store(ns["_pair_values"](None, None, None, None, None, None,
                                                         pa, pb, cols, offsets[0], offsets[1],
                                                         I32(columns), I32(width), I32(16)))
                            else:
                                for mi in range(2):
                                    store([(ok, off, float("-inf")) for ok, off in
                                           ns["_pass_masks"](plans[mi][0], cols, offsets[mi], I32(columns))])
                        else:
                            for mi in range(2):
                                if t0 + mi * 4 >= rows:
                                    continue
                                for pp in plans[mi]:
                                    if not bool(pp.active):
                                        continue
                                    if chunk * 64 < int(pp.vmax):
                                        store(ns["_pass_values"](None, None, None, None, None, pp,
                                                                 cols, offsets[mi], I32(columns),
                                                                 I32(width), I32(16)))
                                    else:
                                        store([(ok, off, float("-inf")) for ok, off in
                                               ns["_pass_masks"](pp, cols, offsets[mi], I32(columns))])
        self.assertEqual(counts, [1] * (rows * columns))
        self.assertEqual(vis_counts, [1] * rows)
        self.assertEqual(rst_counts, [1] * rows)
        for row in range(rows):
            for c in range(columns):
                req = t2r[row]
                valid_req = 0 <= req < requests
                physical = tables[req if valid_req else 0][c // 4]
                expected = valid_req and 0 <= physical < 16 and c < visible[row]
                self.assertEqual(logits[row * columns + c], expected, (row, c))
        return logits

    def test_single_request_shared_tile_and_forced_raw_masks_and_stores_once(self):
        tables = (tuple((p * 3 + 7) % 16 for p in range(33)),)
        for rows in (1, 15, 16, 31, 32, 33, 63, 64, 65):
            visible = tuple((0, 1, 17, 64, 65, 132)[row % 6] for row in range(rows))
            for columns in (1, 3, 17, 61, 64, 65, 131):
                with self.subTest(rows=rows, columns=columns):
                    actual = self.assert_stores((0,) * rows, visible, tables, columns)
                    self.assertEqual(actual, self.assert_stores((0,) * rows, visible, tables,
                                                               columns, force_raw=True))

    def test_mixed_duplicate_invalid_requests_page_holes_and_row_column_tails(self):
        tables = tuple(tuple((7, -1, 16, 3, 0, 15, 2, 0x7FFFFFFF)[(p + req) % 8]
                             for p in range(33)) for req in range(4))
        patterns = ((0, 0, 1, 1, 2, 3, 2, 0), (-1, 0, 4, 1, 2, 2, 3, 3),
                    (0, 1, 2, 3, 0, 1, 2, 3), (1,) * 32 + (2,) * 32)
        for rows in (7, 31, 32, 33, 63, 64, 65):
            for pattern in patterns:
                t2r = tuple(pattern[row % len(pattern)] for row in range(rows))
                visible = tuple((0, 4, 17, 64, 65, 132)[row % 6] for row in range(rows))
                with self.subTest(rows=rows, pattern=pattern):
                    actual = self.assert_stores(t2r, visible, tables, 65)
                    self.assertEqual(actual, self.assert_stores(t2r, visible, tables, 65, force_raw=True))


if __name__ == "__main__":
    unittest.main()

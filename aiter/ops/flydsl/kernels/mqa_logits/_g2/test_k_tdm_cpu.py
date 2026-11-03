# SPDX-License-Identifier: MIT
"""CPU-only byte models of raw and padded bf16 K gathers of [64, 128].

Stock gather carries eight i32 row indices per instruction.  Source rows remain
256 bytes; the padded LDS descriptor skips 4 DWORDs after each 64 data DWORDs,
so eight instructions stage 64 rows with 272-byte stride in 17408 bytes.  The
original 16-KiB row-major model remains a separate, useful raw reference.
Invalid logical or physical pages select physical page zero (not the nearest
valid page); the token's c % 4 is retained, just as in the bf16 WMMA fallback.
Masks are metadata, not zeros or suppressed copies in a staged tile.  The r16
model covers four disjoint per-wave tiles (69632 bytes), four request passes,
and optimized scalar request-dedup/visibility issue guards.  AST checks require
unconditional pre/post pass barriers even when individual waves skip issue.

The independent reference models the fallback's two 16-byte B-fragment loads
per lane and reconstructs the tile with the WMMA operand mapping.  All payloads
are opaque uint16 encodings, including every bf16 NaN payload; no floating-point
conversion, arithmetic, aiter/FlyDSL import, GPU, or third-party dependency.
This is a layout/address model, not a proof of TDM ISA support or lowering.

Run directly, avoiding the GPU-touching aiter package import:
    python aiter/ops/flydsl/kernels/mqa_logits/_g2/test_k_tdm_cpu.py -v
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import random
import struct
import unittest


PAGE_ROWS = 4
TILE_ROWS = 64
HEAD_DIM = 128
ROW_BYTES = HEAD_DIM * 2
PAGE_BYTES = PAGE_ROWS * ROW_BYTES
TILE_BYTES = TILE_ROWS * ROW_BYTES
INDICES_PER_GATHER = 8

# Layout units, not the helper's encoded bitfields.  The stock descriptor API
# takes ELEMENT counts: 128/8 uint16 elements become 64/4 DWORDs (encoding 5/3).
# Padding affects only LDS: source row_width/stride stay 128 elements (256 B).
TDM_PAD_INTERVAL_DWORDS = 64
TDM_PAD_AMOUNT_DWORDS = 4
TDM_K_ROW_BYTES = 272
TDM_K_TILE_BYTES = 17408
PADDING_SENTINEL = bytes.fromhex("a55a3cc396690ff05aa5c33c6996f00f")
R16_WAVES = 4
R16_PASSES = 4
R16_LDS_BYTES = R16_WAVES * TDM_K_TILE_BYTES


@dataclass(frozen=True)
class GatherStart:
    local_row: int
    lds_byte_offset: int
    row_ids: tuple[int, ...]


@dataclass(frozen=True)
class Masks:
    logical_ok: tuple[bool, ...]
    page_ok: tuple[bool, ...]
    col_ok: tuple[bool, ...]
    keep: tuple[bool, ...]


def gather_tile(source, table, origin, columns, visible, req_ok=True):
    """Eight stock gather groups, each copying eight complete source rows."""
    assert source and len(source) % PAGE_BYTES == 0
    assert table and origin >= 0 and origin % TILE_ROWS == 0
    assert 0 < columns <= len(table) * PAGE_ROWS
    num_pages = len(source) // PAGE_BYTES
    logical_masks, page_masks, col_masks, keep_masks = [], [], [], []
    groups = []
    tile = bytearray(TILE_BYTES)
    for group in range(TILE_ROWS // INDICES_PER_GATHER):
        row_ids = []
        for index in range(INDICES_PER_GATHER):
            local_row = group * INDICES_PER_GATHER + index
            c = origin + local_row
            logical = c // PAGE_ROWS
            logical_ok = logical < len(table)
            physical = table[logical if logical_ok else 0]
            page_ok = req_ok and logical_ok and 0 <= physical < num_pages
            physical = physical if page_ok else 0
            row_id = physical * PAGE_ROWS + c % PAGE_ROWS
            row_ids.append(row_id)
            logical_masks.append(logical_ok)
            page_masks.append(page_ok)
            col_masks.append(c < columns)
            keep_masks.append(page_ok and c < visible)
        # The stock interface accepts exactly eight signed i32 row indices.
        encoded_indices = struct.pack("<8i", *row_ids)
        groups.append(struct.unpack("<8i", encoded_indices))
        for index, row_id in enumerate(groups[-1]):
            src = row_id * ROW_BYTES
            dst = (group * INDICES_PER_GATHER + index) * ROW_BYTES
            assert 0 <= src <= len(source) - ROW_BYTES
            tile[dst:dst + ROW_BYTES] = source[src:src + ROW_BYTES]
    return bytes(tile), tuple(groups), Masks(
        tuple(logical_masks), tuple(page_masks), tuple(col_masks), tuple(keep_masks)
    )


def gather_padded_tile(source, table, origin, columns, visible, req_ok=True,
                       *, group_order=None):
    """Model descriptor data-DWORD writes, including skips within each gather.

    Reuse only the raw model's row-index/mask plan; payloads are loaded again
    from source, never from its returned tile.  The skip count resets at each
    eight-row descriptor, whose start is explicitly local_row * 272 bytes.
    """
    _, groups, masks = gather_tile(source, table, origin, columns, visible, req_ok)
    group_order = tuple(range(len(groups))) if group_order is None else tuple(group_order)
    assert sorted(group_order) == list(range(len(groups)))
    tile = bytearray(PADDING_SENTINEL * (TDM_K_TILE_BYTES // len(PADDING_SENTINEL)))
    starts = []
    data_dwords_per_row = ROW_BYTES // 4
    for group in group_order:
        local = group * INDICES_PER_GATHER
        start = local * TDM_K_ROW_BYTES
        starts.append(GatherStart(local, start, groups[group]))
        for data_dw in range(INDICES_PER_GATHER * data_dwords_per_row):
            row_id = groups[group][data_dw // data_dwords_per_row]
            src = row_id * ROW_BYTES + (data_dw % data_dwords_per_row) * 4
            skipped_dw = (data_dw // TDM_PAD_INTERVAL_DWORDS) * TDM_PAD_AMOUNT_DWORDS
            dst = start + (data_dw + skipped_dw) * 4
            assert 0 <= src <= len(source) - 4
            assert start <= dst <= start + INDICES_PER_GATHER * TDM_K_ROW_BYTES - 4
            assert 0 <= dst <= len(tile) - 4
            tile[dst:dst + 4] = source[src:src + 4]
    return bytes(tile), groups, masks, tuple(starts)


def padded_fragment_offsets(nt, s, lane):
    """Actual two-vector LDS consumer addresses, not element reconstruction."""
    assert 0 <= nt < 4 and 0 <= s < 4 and 0 <= lane < 32
    lo = (nt * 16 + lane % 16) * TDM_K_ROW_BYTES + (lane // 16) * 16 + s * 64
    return lo, lo + 32


def padded_tile_fragment(tile, nt, s, lane):
    lo, hi = padded_fragment_offsets(nt, s, lane)
    assert lo % 16 == hi % 16 == 0
    assert 0 <= lo <= len(tile) - 16 and 0 <= hi <= len(tile) - 16
    return tile[lo:lo + 16] + tile[hi:hi + 16]


def fallback_fragments(source, table, origin, columns, visible, req_ok=True):
    """Independent per-lane global-load reference; never calls gather_tile."""
    num_pages = len(source) // PAGE_BYTES
    fragments = {}
    masks = {}
    for nt in range(4):
        for lane in range(32):
            c = origin + nt * 16 + lane % 16
            logical_ok = c // 4 < len(table)
            table_index = c // 4 if logical_ok else 0
            raw_physical = table[table_index]
            page_ok = bool(req_ok and logical_ok and
                           raw_physical >= 0 and raw_physical < num_pages)
            safe_physical = raw_physical if page_ok else 0
            masks[nt, lane] = (logical_ok, page_ok, c < columns,
                               page_ok and c < visible)
            for s in range(4):
                base = (safe_physical * PAGE_BYTES + (c % 4) * ROW_BYTES
                        + (lane // 16) * 16 + s * 64)
                assert 0 <= base and base + 48 <= len(source)
                fragments[nt, s, lane] = (source[base:base + 16]
                                          + source[base + 32:base + 48])
    return fragments, masks


def tile_fragment(tile, nt, s, lane):
    """Read a staged B fragment by the WMMA logical element coordinates."""
    column = nt * 16 + lane % 16
    values = []
    for v in range(16):
        k = s * 32 + (v // 8) * 16 + (lane // 16) * 8 + v % 8
        byte_offset = (column * HEAD_DIM + k) * 2
        values.append(tile[byte_offset:byte_offset + 2])
    return b"".join(values)


class KTileGatherCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One occurrence of every uint16 encoding: all signed zeros, infinities,
        # subnormals, finite values, signaling/quiet NaNs and NaN payloads.
        patterns = list(range(1 << 16))
        random.Random(0xB16).shuffle(patterns)
        cls.source = struct.pack("<65536H", *patterns)
        cls.num_pages = len(cls.source) // PAGE_BYTES

    def assert_equivalent(self, table, origin=0, columns=None, visible=None,
                          req_ok=True):
        columns = len(table) * PAGE_ROWS if columns is None else columns
        visible = columns if visible is None else visible
        tile, groups, masks = gather_tile(
            self.source, table, origin, columns, visible, req_ok
        )
        self.assertEqual(len(tile), 16 * 1024)
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(group) == 8 for group in groups))
        fragments, reference_masks = fallback_fragments(
            self.source, table, origin, columns, visible, req_ok
        )
        restored = bytearray(TILE_BYTES)
        owners = bytearray(TILE_ROWS * HEAD_DIM)
        for (nt, s, lane), fragment in fragments.items():
            self.assertEqual(len(fragment), 32)
            self.assertEqual(tile_fragment(tile, nt, s, lane), fragment,
                             f"B fragment nt={nt}, s={s}, lane={lane}")
            local_col = nt * 16 + lane % 16
            for v in range(16):
                k = s * 32 + (v // 8) * 16 + (lane // 16) * 8 + v % 8
                element = local_col * HEAD_DIM + k
                owners[element] += 1
                restored[element * 2:element * 2 + 2] = fragment[v * 2:v * 2 + 2]
        self.assertEqual(owners, bytearray([1]) * (TILE_ROWS * HEAD_DIM))
        self.assertEqual(tile, bytes(restored), "complete row-major tile bytes")
        for (nt, lane), reference in reference_masks.items():
            local_col = nt * 16 + lane % 16
            self.assertEqual(tuple(mask[local_col] for mask in (
                masks.logical_ok, masks.page_ok, masks.col_ok, masks.keep
            )), reference)
        return tile, groups, masks

    def test_all_uint16_encodings_including_nan_payloads(self):
        table = tuple(range(self.num_pages))
        tiles = [self.assert_equivalent(table, origin)[0]
                 for origin in range(0, self.num_pages * 4, 64)]
        self.assertEqual(b"".join(tiles), self.source)
        bits = struct.unpack("<65536H", b"".join(tiles))
        self.assertEqual(set(bits), set(range(65536)))
        self.assertEqual(sum((b & 0x7F80) == 0x7F80 and (b & 0x7F) != 0
                             for b in bits), 254)

    def test_shuffled_pages_and_aligned_tile_origins(self):
        table = list(range(self.num_pages))
        random.Random(0x64_128).shuffle(table)
        for origin in (0, 64, 128, 192, 256, 320, 384, 448):
            with self.subTest(origin=origin):
                tile, groups, _ = self.assert_equivalent(table, origin)
                self.assertEqual(tuple(row for group in groups for row in group),
                                 tuple(table[c // 4] * 4 + c % 4
                                       for c in range(origin, origin + 64)))
                expected = b"".join(self.source[p * PAGE_BYTES:(p + 1) * PAGE_BYTES]
                                    for p in table[origin // 4:origin // 4 + 16])
                self.assertEqual(tile, expected)

    def test_four_waves_two_eight_row_groups_each(self):
        table = tuple((i * 29 + 7) % self.num_pages for i in range(32))
        for origin in (0, 64):
            _, groups, _ = self.assert_equivalent(table, origin)
            owners = [0] * TILE_ROWS
            for wave in range(4):
                for group in range(2):
                    local = wave * 16 + group * 8
                    expected = []
                    # Each eight-row instruction needs two page lookups, with
                    # four row ids expanded from each page, as in the producer.
                    for page in range(2):
                        c = origin + local + page * 4
                        physical = table[c // 4]
                        expected.extend(physical * 4 + offset for offset in range(4))
                    self.assertEqual(groups[wave * 2 + group], tuple(expected))
                    for row in range(local, local + 8):
                        owners[row] += 1
            self.assertEqual(owners, [1] * TILE_ROWS)

    def test_invalid_physical_pages_select_zero_not_last_page(self):
        table = [7, -1, self.num_pages, self.num_pages + 17,
                 0x7FFFFFFF, -0x80000000, 0, self.num_pages - 1] * 4
        for origin in (0, 64):
            with self.subTest(origin=origin):
                tile, groups, masks = self.assert_equivalent(table, origin)
                row_ids = sum(groups, ())
                for local_col in range(64):
                    c = origin + local_col
                    valid = 0 <= table[c // 4] < self.num_pages
                    self.assertEqual(masks.page_ok[local_col], valid)
                    if not valid:
                        self.assertEqual(row_ids[local_col], c % 4)
                        start = (c % 4) * ROW_BYTES
                        self.assertEqual(tile[local_col * ROW_BYTES:(local_col + 1) * ROW_BYTES],
                                         self.source[start:start + ROW_BYTES])

    def test_width_and_column_tails_and_invalid_logical_pages(self):
        for width in (1, 3, 15, 16, 17, 19, 31, 33, 65):
            # table[0] deliberately nonzero: invalid logical pages must still
            # select physical zero after the safe table-index-zero lookup.
            table = tuple((i * 29 + 7) % self.num_pages for i in range(width))
            for columns in sorted({1, max(1, width * 4 - 3), width * 4}):
                origins = {0, ((columns - 1) // 64) * 64,
                           ((width * 4 + 63) // 64) * 64}
                for origin in sorted(origins):
                    with self.subTest(width=width, columns=columns, origin=origin):
                        _, groups, masks = self.assert_equivalent(table, origin, columns)
                        row_ids = sum(groups, ())
                        for local_col in range(64):
                            c = origin + local_col
                            self.assertEqual(masks.logical_ok[local_col], c // 4 < width)
                            self.assertEqual(masks.col_ok[local_col], c < columns)
                            if c // 4 >= width:
                                self.assertEqual(row_ids[local_col], c % 4)
                                self.assertFalse(masks.page_ok[local_col])

    def test_masks_do_not_zero_payload_or_conflate_store_and_keep(self):
        table = (7, -1, self.num_pages, 3) * 4
        tile, groups, _ = self.assert_equivalent(table, columns=61, visible=64)
        for visible in (0, 1, 4, 17, 60, 61, 64):
            with self.subTest(visible=visible):
                other, other_groups, masks = self.assert_equivalent(
                    table, columns=61, visible=visible
                )
                self.assertEqual(other, tile)
                self.assertEqual(other_groups, groups)
                self.assertEqual(masks.keep, tuple(
                    masks.page_ok[c] and c < visible for c in range(64)
                ))
                self.assertEqual(masks.col_ok, tuple(c < 61 for c in range(64)))
                if visible == 64:
                    self.assertTrue(masks.keep[63])
                    self.assertFalse(masks.col_ok[63])
        _, invalid_groups, masks = self.assert_equivalent(table, req_ok=False)
        self.assertEqual(sum(invalid_groups, ()), tuple(c % 4 for c in range(64)))
        self.assertFalse(any(masks.page_ok))
        self.assertFalse(any(masks.keep))
        self.assertTrue(all(masks.logical_ok))
        self.assertTrue(all(masks.col_ok))


class KPaddedTileGatherCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        KTileGatherCPU.setUpClass()
        cls.source = KTileGatherCPU.source
        cls.num_pages = KTileGatherCPU.num_pages

    def assert_equivalent(self, table, origin=0, columns=None, visible=None,
                          req_ok=True, group_order=None):
        columns = len(table) * PAGE_ROWS if columns is None else columns
        visible = columns if visible is None else visible
        tile, groups, masks, starts = gather_padded_tile(
            self.source, table, origin, columns, visible, req_ok,
            group_order=group_order,
        )
        self.assertEqual(len(tile), TDM_K_TILE_BYTES)
        self.assertEqual(len(groups), 8)
        self.assertEqual(len(starts), 8)
        self.assertEqual(sorted(start.local_row for start in starts), list(range(0, 64, 8)))
        for start in starts:
            self.assertEqual(start.lds_byte_offset, start.local_row * 272)
            self.assertEqual(start.row_ids, groups[start.local_row // 8])
            self.assertEqual(len(struct.pack("<8i", *start.row_ids)), 32)
            self.assertEqual(start.lds_byte_offset % 16, 0)
            self.assertLessEqual(start.lds_byte_offset + 8 * 272, len(tile))

        # This reference never calls either gather model and loads opaque bytes
        # directly from the source page selected independently for every lane.
        fragments, reference_masks = fallback_fragments(
            self.source, table, origin, columns, visible, req_ok,
        )
        restored = bytearray(TILE_BYTES)
        owners = bytearray(TILE_ROWS * HEAD_DIM)
        for (nt, s, lane), fragment in fragments.items():
            lo, hi = padded_fragment_offsets(nt, s, lane)
            row = nt * 16 + lane % 16
            row_start = row * TDM_K_ROW_BYTES
            self.assertEqual(hi - lo, 32)
            for address in (lo, hi):
                self.assertEqual(address % 16, 0)
                self.assertGreaterEqual(address, row_start)
                self.assertLessEqual(address + 16, row_start + ROW_BYTES)
                self.assertLessEqual(address + 16, len(tile))
            self.assertEqual(padded_tile_fragment(tile, nt, s, lane), fragment,
                             f"padded B fragment nt={nt}, s={s}, lane={lane}")
            for v in range(16):
                # Independent WMMA logical coordinates, rather than reusing the
                # consumer's two-vector address expression to reconstruct data.
                k = s * 32 + (v // 8) * 16 + (lane // 16) * 8 + v % 8
                element = row * HEAD_DIM + k
                owners[element] += 1
                restored[element * 2:element * 2 + 2] = fragment[v * 2:v * 2 + 2]
        self.assertEqual(owners, bytearray([1]) * (TILE_ROWS * HEAD_DIM))
        raw_payload = b"".join(tile[row * 272:row * 272 + 256] for row in range(64))
        self.assertEqual(raw_payload, bytes(restored), "all padded payload bytes vs raw loads")
        for row in range(64):
            self.assertEqual(tile[row * 272 + 256:(row + 1) * 272], PADDING_SENTINEL,
                             f"skipped padding changed at row {row}")
        for (nt, lane), reference in reference_masks.items():
            row = nt * 16 + lane % 16
            self.assertEqual(tuple(mask[row] for mask in (
                masks.logical_ok, masks.page_ok, masks.col_ok, masks.keep,
            )), reference)
        return tile, groups, masks, starts, raw_payload

    def test_all_uint16_encodings_byte_exact_including_nans_and_subnormals(self):
        table = tuple(range(self.num_pages))
        payloads = [self.assert_equivalent(table, origin)[4]
                    for origin in range(0, self.num_pages * PAGE_ROWS, TILE_ROWS)]
        restored = b"".join(payloads)
        self.assertEqual(restored, self.source)
        bits = struct.unpack("<65536H", restored)
        self.assertEqual(set(bits), set(range(65536)))
        self.assertEqual(sum((b & 0x7F80) == 0x7F80 and (b & 0x7F) != 0
                             for b in bits), 254)
        self.assertEqual(sum((b & 0x7F80) == 0 and (b & 0x7F) != 0 for b in bits), 254)
        self.assertEqual(sum((b & 0x7FFF) == 0 for b in bits), 2)
        self.assertEqual(sum((b & 0x7FFF) == 0x7F80 for b in bits), 2)

    def test_all_nt_s_lane_vectors_are_aligned_in_bounds_and_never_read_padding(self):
        readers = bytearray(TDM_K_TILE_BYTES)
        vector_count = 0
        max_end = 0
        for nt in range(4):
            for s in range(4):
                for lane in range(32):
                    lo, hi = padded_fragment_offsets(nt, s, lane)
                    self.assertEqual(hi, lo + 32)
                    for address in (lo, hi):
                        self.assertEqual(address % 16, 0)
                        self.assertGreaterEqual(address, 0)
                        self.assertLessEqual(address + 16, TDM_K_TILE_BYTES)
                        self.assertLessEqual(address % 272 + 16, 256)
                        for byte in range(address, address + 16):
                            readers[byte] += 1
                        vector_count += 1
                        max_end = max(max_end, address + 16)
        self.assertEqual(vector_count, 4 * 4 * 32 * 2)
        self.assertEqual(max_end, TDM_K_TILE_BYTES - len(PADDING_SENTINEL))
        self.assertEqual(readers, (bytes([1]) * 256 + bytes(16)) * 64)

    def test_logical_64_bank_mapping_spreads_each_half_wave_over_all_banks(self):
        # Logical DWORD-bank mapping only, NOT hardware conflict/timing proof.
        for nt in range(4):
            for s in range(4):
                for extra in (0, 32):
                    padded_wave_hits = [0] * 64
                    for kgrp in (0, 1):
                        raw_hits = [0] * 64
                        padded_hits = [0] * 64
                        for n in range(16):
                            row = nt * 16 + n
                            fragment_offset = s * 64 + kgrp * 16 + extra
                            raw = row * 256 + fragment_offset
                            padded = row * 272 + fragment_offset
                            for dw in range(4):
                                raw_hits[(raw // 4 + dw) % 64] += 1
                                bank = (padded // 4 + dw) % 64
                                padded_hits[bank] += 1
                                padded_wave_hits[bank] += 1
                        self.assertEqual(sorted(raw_hits), [0] * 60 + [16] * 4)
                        self.assertEqual(padded_hits, [1] * 64)
                    self.assertEqual(padded_wave_hits, [2] * 64)

    def test_eight_row_gather_starts_and_padding_under_varied_producer_orders(self):
        table = list(range(self.num_pages))
        random.Random(0x272).shuffle(table)
        orders = (tuple(range(8)), tuple(reversed(range(8))),
                  (0, 2, 4, 6, 1, 3, 5, 7), (6, 3, 0, 7, 2, 5, 1, 4))
        for origin in (0, 64, 448):
            baseline = None
            for order in orders:
                with self.subTest(origin=origin, group_order=order):
                    tile, groups, _, starts, payload = self.assert_equivalent(
                        table, origin, group_order=order,
                    )
                    self.assertEqual(tuple(s.lds_byte_offset for s in starts),
                                     tuple(group * 8 * 272 for group in order))
                    raw, raw_groups, _ = gather_tile(
                        self.source, table, origin, len(table) * 4, len(table) * 4,
                    )
                    self.assertEqual(groups, raw_groups)
                    self.assertEqual(payload, raw, "raw row-major model remains useful")
                    baseline = tile if baseline is None else baseline
                    self.assertEqual(tile, baseline, "disjoint producer writes commute")

    def test_invalid_pages_and_column_tails_still_copy_raw_page_zero_bytes(self):
        table = (7, -1, self.num_pages, 0x7FFFFFFF, -0x80000000, 3, 0) * 3
        for origin in (0, 64, 128):
            for columns in (1, 61, len(table) * PAGE_ROWS):
                with self.subTest(origin=origin, columns=columns):
                    _, groups, masks, _, _ = self.assert_equivalent(
                        table, origin, columns=columns, visible=17,
                    )
                    for row, row_id in enumerate(sum(groups, ())):
                        c = origin + row
                        valid = c // 4 < len(table) and 0 <= table[c // 4] < self.num_pages
                        self.assertEqual(masks.page_ok[row], valid)
                        if not valid:
                            self.assertEqual(row_id, c % 4)
        _, groups, masks, _, _ = self.assert_equivalent(table, req_ok=False)
        self.assertEqual(sum(groups, ()), tuple(c % 4 for c in range(64)))
        self.assertFalse(any(masks.page_ok))
        self.assertFalse(any(masks.keep))

    def test_masks_neither_zero_data_nor_change_skipped_padding(self):
        table = (7, -1, self.num_pages, 3) * 4
        baseline = self.assert_equivalent(table, columns=61, visible=64)[0]
        for visible in (0, 1, 4, 17, 60, 61, 64):
            with self.subTest(visible=visible):
                tile, _, masks, _, _ = self.assert_equivalent(table, columns=61, visible=visible)
                self.assertEqual(tile, baseline)
                self.assertEqual(masks.keep, tuple(masks.page_ok[c] and c < visible
                                                  for c in range(64)))
                self.assertEqual(masks.col_ok, tuple(c < 61 for c in range(64)))
                if visible == 64:
                    self.assertTrue(masks.keep[63])
                    self.assertFalse(masks.col_ok[63])


@dataclass(frozen=True)
class R16WavePass:
    wave: int
    request: int
    request_ok: bool
    active: bool
    visible: int
    issued: bool
    starts: tuple[GatherStart, ...]
    masks: Masks


def stage_r16_pass(source, tables, t2r, rblk, origin, columns, row_visible, p,
                   *, scratch=None, wave_order=None, active_only=False):
    """One full r16 pass: unconditional byte proof or optimized active subset.

    active_only=True models the SGPR request-dedup/visibility issue guard; the
    default retains the stronger all-four-passes layout proof, even for bytes
    that optimized hardware need not stage or consume.

    Each wave's four query rows may have four different requests.  A descriptor
    uses row t0+p's scalar request, clamps invalid requests to request-table zero
    for safe metadata lookup, and selects physical page zero if req_ok is false.
    Only payload bytes are written, preserving preexisting padding canaries.
    """
    assert 0 <= p < R16_PASSES and rblk >= 0
    assert rblk * 16 + 16 <= len(t2r) == len(row_visible)
    assert tables and all(len(table) == len(tables[0]) for table in tables)
    scratch = bytearray(PADDING_SENTINEL * (R16_LDS_BYTES // 16)
                        if scratch is None else scratch)
    assert len(scratch) == R16_LDS_BYTES
    wave_order = tuple(range(4)) if wave_order is None else tuple(wave_order)
    assert sorted(wave_order) == list(range(4))
    passes = []
    for wave in wave_order:
        t0 = rblk * 16 + wave * 4
        req = t2r[t0 + p]
        req_ok = 0 <= req < len(tables)
        safe_req = req if req_ok else 0
        active = req not in t2r[t0:t0 + p]
        visible = max((row_visible[t0 + i] for i in range(4)
                       if req_ok and t2r[t0 + i] == req), default=0)
        tile, _, masks, starts = gather_padded_tile(
            source, tables[safe_req], origin, columns, visible, req_ok,
        )
        base = wave * TDM_K_TILE_BYTES
        issue = not active_only or (req_ok and active and origin < visible)
        absolute_starts = tuple(GatherStart(start.local_row, base + start.lds_byte_offset,
                                           start.row_ids) for start in starts) if issue else ()
        if issue:
            for row in range(64):
                dst = base + row * TDM_K_ROW_BYTES
                assert base <= dst and dst + ROW_BYTES <= base + TDM_K_TILE_BYTES
                assert dst + ROW_BYTES <= len(scratch)
                local = row * TDM_K_ROW_BYTES
                scratch[dst:dst + ROW_BYTES] = tile[local:local + ROW_BYTES]
        passes.append(R16WavePass(wave, req, req_ok, active, visible, issue, absolute_starts, masks))
    return bytes(scratch), tuple(passes)


class KR16PerWaveGatherCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        KTileGatherCPU.setUpClass()
        cls.source = KTileGatherCPU.source
        cls.num_pages = KTileGatherCPU.num_pages
        cls.tables = tuple(tuple((request * 13 + logical * 29 + 7) % cls.num_pages
                                 for logical in range(33)) for request in range(8))

    def assert_pass(self, tables, t2r, p, *, origin=0, columns=None, row_visible=None,
                    scratch=None, wave_order=None, rblk=0, active_only=False):
        columns = len(tables[0]) * 4 if columns is None else columns
        row_visible = [columns] * len(t2r) if row_visible is None else row_visible
        before = (PADDING_SENTINEL * (R16_LDS_BYTES // 16)) if scratch is None else scratch
        actual, passes = stage_r16_pass(
            self.source, tables, t2r, rblk, origin, columns, row_visible, p,
            scratch=scratch, wave_order=wave_order, active_only=active_only,
        )
        self.assertEqual(len(actual), 69632)
        self.assertEqual(len(passes), 4)
        self.assertEqual(sorted(item.wave for item in passes), list(range(4)))
        self.assertEqual(sum(len(item.starts) for item in passes),
                         sum(8 for item in passes if item.issued))
        for item in passes:
            wave = item.wave
            base = wave * 17408
            t0 = rblk * 16 + wave * 4
            req = t2r[t0 + p]
            req_ok = 0 <= req < len(tables)
            self.assertEqual((item.request, item.request_ok), (req, req_ok))
            self.assertEqual(item.active, all(t2r[t0 + prior] != req for prior in range(p)))
            expected_visible = max((row_visible[t0 + i] for i in range(4)
                                    if req_ok and t2r[t0 + i] == req), default=0)
            self.assertEqual(item.visible, expected_visible)
            self.assertEqual(item.issued, not active_only or
                             (req_ok and item.active and origin < expected_visible))
            if not item.issued:
                self.assertEqual(item.starts, ())
                self.assertEqual(actual[base:base + 17408], before[base:base + 17408],
                                 "inactive/hidden wave must leave its entire tile untouched")
                continue
            table = tables[req if req_ok else 0]
            fragments, masks = fallback_fragments(
                self.source, table, origin, columns, expected_visible, req_ok,
            )
            self.assertEqual(tuple(s.lds_byte_offset for s in item.starts),
                             tuple(base + group * 8 * 272 for group in range(8)))
            for start in item.starts:
                self.assertEqual(start.lds_byte_offset % 16, 0)
                self.assertGreaterEqual(start.lds_byte_offset, base)
                self.assertLessEqual(start.lds_byte_offset + 8 * 272, base + 17408)
                self.assertLessEqual(start.lds_byte_offset + 8 * 272, 69632)
            for (nt, s, lane), expected in fragments.items():
                lo, hi = padded_fragment_offsets(nt, s, lane)
                self.assertEqual(hi - lo, 32)
                for offset in (base + lo, base + hi):
                    self.assertEqual(offset % 16, 0)
                    self.assertGreaterEqual(offset, base)
                    self.assertLessEqual(offset + 16, base + 17408)
                    self.assertLessEqual(offset + 16, 69632)
                self.assertEqual(actual[base + lo:base + lo + 16]
                                 + actual[base + hi:base + hi + 16], expected,
                                 f"wave={wave}, pass={p}, nt={nt}, s={s}, lane={lane}")
            for (nt, lane), expected_masks in masks.items():
                row = nt * 16 + lane % 16
                self.assertEqual(tuple(mask[row] for mask in (
                    item.masks.logical_ok, item.masks.page_ok, item.masks.col_ok, item.masks.keep,
                )), expected_masks)
            for row in range(64):
                pad = base + row * 272 + 256
                self.assertEqual(actual[pad:pad + 16], before[pad:pad + 16],
                                 f"wave={wave}, pass={p}, row={row} padding overwritten")
        return actual, passes

    def test_all_four_passes_keep_mixed_request_wave_tiles_separate(self):
        t2r = (0, 1, 2, 3, 4, 3, 2, 1, 7, 5, 4, 6, 2, 6, 0, 5)
        scratch = None
        for p in range(4):
            with self.subTest(p=p):
                scratch, _ = self.assert_pass(self.tables, t2r, p, origin=64, scratch=scratch)
                # These request mappings differ at each pass, so reusing another
                # wave's tile (or stale previous-pass bytes) cannot pass equality.
                payloads = [b"".join(scratch[wave * 17408 + row * 272:
                                             wave * 17408 + row * 272 + 256]
                                      for row in range(64)) for wave in range(4)]
                self.assertEqual(len(set(payloads)), len({t2r[wave * 4 + p] for wave in range(4)}))

    def test_invalid_requests_physical_pages_and_logical_tails_select_page_zero(self):
        tables = tuple(tuple((7, -1, self.num_pages, 0x7FFFFFFF, 3)[(i + req) % 5]
                             for i in range(17)) for req in range(3))
        t2r = (-1, 0, 3, 4, 2, -0x80000000, 1, 0x7FFFFFFF, 0, 1, 2, 3, 4, -1, 1, 2)
        for origin in (0, 64, 128):
            for p in range(4):
                with self.subTest(origin=origin, p=p):
                    _, passes = self.assert_pass(tables, t2r, p, origin=origin, columns=65)
                    for item in passes:
                        if not item.request_ok:
                            self.assertEqual(sum((s.row_ids for s in item.starts), ()),
                                             tuple((origin + row) % 4 for row in range(64)))
                            self.assertFalse(any(item.masks.page_ok))
                            self.assertFalse(any(item.masks.keep))

    def test_unconditional_layout_proof_covers_deduplicated_and_invisible_passes(self):
        t2r = (0,) * 4 + (1,) * 4 + (-1,) * 4 + (2,) * 4
        scratch = None
        for p in range(4):
            with self.subTest(p=p):
                scratch, passes = self.assert_pass(self.tables, t2r, p,
                                                  row_visible=[0] * 16, scratch=scratch)
                self.assertEqual(tuple(item.active for item in passes), (p == 0,) * 4)
                self.assertTrue(all(item.visible == 0 for item in passes))
                self.assertEqual(sum(len(item.starts) for item in passes), 32)

    def test_optimized_active_subset_skips_invalid_duplicate_and_invisible_waves(self):
        t2r = (0, 0, 1, 1, 2, 3, 2, 4, -1, 0, 8, 1, 5, 6, 7, 5)
        visible = (65, 65, 132, 132, 64, 0, 64, 132, 132, 132, 132, 132, 132, 0, 132, 132)
        scratch = bytes([0xD3]) * 69632
        for p in range(4):
            with self.subTest(p=p):
                scratch, passes = self.assert_pass(self.tables, t2r, p, origin=64,
                                                  row_visible=visible, scratch=scratch,
                                                  active_only=True)
                expected = []
                for wave in range(4):
                    req = t2r[wave * 4 + p]
                    owned_vis = max((visible[wave * 4 + i] for i in range(4)
                                     if t2r[wave * 4 + i] == req), default=0)
                    expected.append(0 <= req < 8 and req not in t2r[wave * 4:wave * 4 + p]
                                    and 64 < owned_vis)
                self.assertEqual(tuple(item.issued for item in passes), tuple(expected))

    def test_wave_schedules_preserve_arbitrary_padding_across_passes_and_blocks(self):
        rng = random.Random(0x69632)
        canaries = bytes(rng.randrange(256) for _ in range(69632))
        t2r = tuple((row * 3 + row // 4) % 8 for row in range(48))
        orders = ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2))
        for rblk in (0, 1, 2):
            scratch = canaries
            for p in range(4):
                with self.subTest(rblk=rblk, p=p):
                    baseline = None
                    for order in orders:
                        actual, _ = self.assert_pass(self.tables, t2r, p, rblk=rblk,
                                                     scratch=scratch, wave_order=order)
                        baseline = actual if baseline is None else baseline
                        self.assertEqual(actual, baseline, "per-wave writes must commute")
                    scratch = baseline


def _integer_expression(node, values):
    """Restricted AST arithmetic evaluation; never import/execute kernel code."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp):
        left = _integer_expression(node.left, values)
        right = _integer_expression(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise AssertionError("unsupported layout expression: " + ast.unparse(node))


def _calls(root, name):
    return [node for node in ast.walk(root) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


class KPaddedKernelSourceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
        cls.tree = ast.parse(path.read_text(encoding="utf-8"))
        cls.build = next(node for node in cls.tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "_build_wmma_kernel")
        cls.helpers = {node.name: node for node in cls.build.body if isinstance(node, ast.FunctionDef)}
        cls.values = {"HEAD_DIM": HEAD_DIM, "PAGE_SIZE": PAGE_ROWS, "WMMA_COLS": TILE_ROWS,
                      "WMMA_K": 32, "WMMA_N": 16, "WMMA_N_TILES": 4,
                      "TDM_K_ROW_BYTES": TDM_K_ROW_BYTES,
                      "TDM_K_TILE_BYTES": TDM_K_TILE_BYTES,
                      "TDM_PAD_INTERVAL_DWORDS": TDM_PAD_INTERVAL_DWORDS,
                      "TDM_PAD_AMOUNT_DWORDS": TDM_PAD_AMOUNT_DWORDS}

    def test_kernel_layout_constants_match_byte_model(self):
        for name, expected in self.values.items():
            assignments = [node for node in self.tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == name
                                   for target in node.targets)]
            self.assertEqual(len(assignments), 1, f"missing/duplicate constant {name}")
            self.assertEqual(_integer_expression(assignments[0].value, self.values), expected, name)
        self.assertEqual(TDM_PAD_INTERVAL_DWORDS * 4, ROW_BYTES)
        self.assertEqual(ROW_BYTES + TDM_PAD_AMOUNT_DWORDS * 4, TDM_K_ROW_BYTES)
        self.assertEqual(TILE_ROWS * TDM_K_ROW_BYTES, TDM_K_TILE_BYTES)

    def test_descriptor_keeps_source_256b_but_converts_dword_padding_to_elements(self):
        gather = next(node for node in self.tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_tdm_k_gather")
        descriptors = _calls(gather, "tdm_ops.make_tensor_gather_descriptor")
        self.assertEqual(len(descriptors), 1)
        keywords = {keyword.arg: keyword.value for keyword in descriptors[0].keywords}
        expected = {"row_width": 128, "tensor_dim0": 128, "stride": 128,
                    "elem_bytes": 2, "index_size": 32, "pad_interval": 128, "pad_amount": 8}
        for name, value in expected.items():
            self.assertEqual(_integer_expression(keywords[name], self.values), value, name)
        for keyword, constant in (("pad_interval", "TDM_PAD_INTERVAL_DWORDS"),
                                  ("pad_amount", "TDM_PAD_AMOUNT_DWORDS")):
            self.assertTrue(any(isinstance(node, ast.Name) and node.id == constant
                                for node in ast.walk(keywords[keyword])), keyword)
        self.assertEqual(ast.unparse(keywords["lds_byte_offset"]), "fx.Index(byte_off)")

    def test_allocator_and_every_eight_row_gather_use_padded_bytes(self):
        storage = self.helpers["_pair_storage_tdm"]
        arrays = [node for node in ast.walk(storage) if isinstance(node, ast.Subscript)
                  and ast.unparse(node.value) == "fx.Array"]
        self.assertEqual(len(arrays), 1)
        self.assertIsInstance(arrays[0].slice, ast.Tuple)
        element_type, count, alignment = arrays[0].slice.elts
        self.assertEqual(ast.unparse(element_type), "fx.Int32")
        self.assertEqual(_integer_expression(count, self.values) * 4, 17408)
        self.assertGreaterEqual(_integer_expression(alignment, self.values), 16)
        allocators = _calls(storage, "fx.SharedAllocator")
        self.assertEqual(len(allocators), 1)
        allocator_keywords = {k.arg: k.value for k in allocators[0].keywords}
        self.assertGreaterEqual(_integer_expression(allocator_keywords["base_alignment"], self.values), 16)
        self.assertIs(allocator_keywords["static"].value, True)
        stage = self.helpers["_pair_stage_tdm"]
        gathers = _calls(stage, "_tdm_k_gather")
        self.assertEqual(len(gathers), 1)
        locals_ = [node.value for node in ast.walk(stage) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "local"
                           for target in node.targets)]
        self.assertEqual(len(locals_), 1)
        group_loops = [node for node in ast.walk(stage) if isinstance(node, ast.For)
                       and isinstance(node.target, ast.Name) and node.target.id == "group"]
        self.assertEqual(len(group_loops), 1)
        self.assertEqual(ast.unparse(group_loops[0].iter), "range_constexpr(2)")
        for wave in range(4):
            for group in range(2):
                local = wave * 16 + group * 8
                self.assertEqual(_integer_expression(locals_[0], dict(
                    self.values, wave_id=wave, group=group)), local)
                self.assertEqual(_integer_expression(gathers[0].args[-1], dict(self.values, local=local)),
                                 local * 272)

    def test_padded_consumer_base_and_lo_hi_vectors_keep_raw_source_stride(self):
        padded = self.helpers["_pair_base_tdm"]
        returns = [node for node in ast.walk(padded) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        for tdm_cols in (64, 32):   # r32/legacy-r16 tile and the R16 c32 tile
            for c in range(128):
                for kgrp in (0, 1):
                    values = dict(self.values, c=c, kgrp=kgrp, phys=7, off=3, tdm_cols=tdm_cols,
                                  key_row_bytes=256, page_bytes=1024)
                    self.assertEqual(_integer_expression(returns[0].value, values),
                                     (c % tdm_cols) * 272 + kgrp * 16)
        fragment = self.helpers["_pair_frag_tdm"]
        loads = _calls(fragment, "fx.ptr_load")
        self.assertEqual(len(loads), 2)
        for load, extra in zip(loads, (0, 32)):
            for nt in range(4):
                for s in range(4):
                    for lane in range(32):
                        byte_off = padded_fragment_offsets(nt, s, lane)[0]
                        self.assertEqual(_integer_expression(load.args[0],
                                                             dict(self.values, ptr=0, byte_off=byte_off)),
                                         (byte_off + extra) // 4)
            types = [k.value for k in load.keywords if k.arg == "result_type"]
            self.assertEqual(len(types), 1)
            self.assertEqual(ast.unparse(types[0]), "Vec.make_type(4, fx.Int32)")
        raw = self.helpers["_pair_base_raw"]
        raw_return = next(node for node in ast.walk(raw) if isinstance(node, ast.Return))
        for phys in (0, 7, 127):
            for off in range(4):
                for kgrp in (0, 1):
                    self.assertEqual(_integer_expression(raw_return.value, dict(
                        self.values, phys=phys, off=off, c=63, kgrp=kgrp,
                        key_row_bytes=256, page_bytes=1024)),
                        phys * 1024 + off * 256 + kgrp * 16)
        assignments = {target.id: node.value for node in self.build.body
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        self.assertEqual(_integer_expression(assignments["key_row_bytes"], self.values), 256)
        self.assertEqual(_integer_expression(assignments["kstep_bytes"], self.values), 64)


class KR16PerWaveKernelSourceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        KPaddedKernelSourceCPU.setUpClass()
        cls.tree = KPaddedKernelSourceCPU.tree
        cls.build = KPaddedKernelSourceCPU.build
        cls.helpers = KPaddedKernelSourceCPU.helpers
        cls.values = dict(KPaddedKernelSourceCPU.values, WMMA_WAVES=4,
                          WMMA_ROWS_PER_WAVE=4, WMMA_PASSES=4,
                          WMMA_WAVE=32, WMMA_THREADS=128)
        cls.stage = cls.helpers["_r16_stage_tdm"]
        cls.kernel = cls.helpers["kernel_tdm_r16"]

    def test_one_69632b_allocator_and_all_wave_gather_offsets_are_disjoint(self):
        # Round R16: the same source serves the legacy c64 (69,632 B) and the
        # default forced-TDM c32 (34,816 B) geometry; both are proven here.
        storage = self.helpers["_r16_storage_tdm"]
        self.assertEqual(len(_calls(storage, "fx.SharedAllocator")), 1)
        self.assertFalse(any(isinstance(node, (ast.If, ast.For, ast.While))
                             for node in ast.walk(storage)))
        arrays = [node for node in ast.walk(storage) if isinstance(node, ast.Subscript)
                  and ast.unparse(node.value) == "fx.Array"]
        self.assertEqual(len(arrays), 1)
        element_type, count, alignment = arrays[0].slice.elts
        self.assertEqual(ast.unparse(element_type), "fx.Int32")
        build = {target.id: node.value for node in self.build.body
                 if isinstance(node, ast.Assign) for target in node.targets
                 if isinstance(target, ast.Name)}
        self.assertEqual(ast.unparse(build["tdm_cols"]),
                         "R16_TDM_COLS if rows_per_block == 16 else WMMA_COLS")
        self.assertEqual(ast.unparse(build["cols_per_block"]),
                         "tdm_cols if use_tdm else WMMA_COLS")
        assignments = {target.id: node.value for node in ast.walk(self.stage)
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        loops = [node for node in ast.walk(self.stage) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "group"]
        self.assertEqual(len(loops), 1)
        self.assertEqual(ast.unparse(loops[0].iter), "range_constexpr(tdm_cols // 8)")
        gathers = _calls(self.stage, "_tdm_k_gather")
        self.assertEqual(len(gathers), 1)
        self.assertEqual(ast.unparse(gathers[0].args[-1]), "byte_off")
        self.assertEqual(len(_calls(self.kernel, "_r16_storage_tdm")), 1)
        kernel_assignments = {target.id: node.value for node in self.kernel.body
                              if isinstance(node, ast.Assign) for target in node.targets
                              if isinstance(target, ast.Name)}
        for tdm_cols, lds_bytes in ((64, 69632), (32, 34816)):
            tile_bytes = _integer_expression(build["tdm_tile_bytes"],
                                             dict(self.values, tdm_cols=tdm_cols))
            self.assertEqual(tile_bytes, tdm_cols * 272)
            self.assertEqual(tile_bytes % 128, 0)
            values = dict(self.values, tdm_cols=tdm_cols, tdm_tile_bytes=tile_bytes)
            self.assertEqual(_integer_expression(count, values) * 4, lds_bytes)
            self.assertGreaterEqual(_integer_expression(alignment, values), 128)
            groups = _integer_expression(loops[0].iter.args[0], values)
            self.assertEqual(groups, tdm_cols // 8)
            ranges = []
            for wave in range(4):
                for group in range(groups):
                    local = _integer_expression(assignments["local"], dict(values, group=group))
                    self.assertEqual(local, group * 8)
                    offset = _integer_expression(assignments["byte_off"], dict(
                        values, wave_id=wave, local=local))
                    self.assertEqual(offset, wave * tile_bytes + local * 272)
                    self.assertEqual(offset % 128, 0)
                    self.assertLessEqual(offset + 8 * 272, lds_bytes)
                    ranges.append((offset, offset + 8 * 272))
            self.assertEqual(ranges, [(g * 2176, (g + 1) * 2176) for g in range(4 * groups)])
            for wave in range(4):
                self.assertEqual(_integer_expression(kernel_assignments["kptr"], dict(
                    values, storage=0, wave=wave)) * 4, wave * tile_bytes)
            for chunk in (0, 1, 7):
                col0 = _integer_expression(kernel_assignments["col0"], dict(values, chunk=chunk))
                self.assertEqual(col0, chunk * tdm_cols)
        self.assertEqual(ast.unparse(kernel_assignments["cols"]),
                         "_lane_cols(col0, n_in, tdm_cols // WMMA_N)")

    def test_producer_request_page_metadata_are_wave_uniform_and_safely_clamped(self):
        assignments = {target.id: node.value for node in ast.walk(self.stage)
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        self.assertEqual(ast.unparse(assignments["wave_id"]), "fx.Int32(fx.rocdl.wave_id())")
        self.assertEqual(ast.unparse(assignments["req_ok"]), "(req >= 0) & (req < num_requests)")
        self.assertEqual(ast.unparse(assignments["safe_req"]), "req_ok.select(req, 0)")
        self.assertEqual(ast.unparse(assignments["safe_logical"]), "logical_ok.select(logical, 0)")
        self.assertEqual(ast.unparse(assignments["valid"]),
                         "req_ok & logical_ok & (phys >= 0) & (phys < num_pages)")
        self.assertEqual(ast.unparse(assignments["phys"]), "valid.select(phys, 0)")
        loads = _calls(self.stage, "_ld_uniform_i32")
        self.assertEqual(len(loads), 2)
        self.assertEqual([ast.unparse(load.args[0]) for load in loads], ["t2r_rsrc", "pt_rsrc"])
        self.assertFalse(_calls(self.stage, "_ld_i32"))
        self.assertFalse(any(isinstance(node, ast.Name) and node.id in
                             ("lane", "kgrp", "pp", "plans", "n_in") for node in ast.walk(self.stage)))
        for rblk in (0, 1, 17):
            for wave in range(4):
                t0 = _integer_expression(assignments["t0"], dict(self.values, rblk=rblk, wave_id=wave))
                self.assertEqual(t0, rblk * 16 + wave * 4)
                for p in range(4):
                    self.assertEqual(_integer_expression(loads[0].args[1], dict(self.values, t0=t0, p=p)),
                                     (rblk * 16 + wave * 4 + p) * 4)
        for width in (1, 17, 33):
            for safe_req in (0, 1, 7):
                for safe_logical in range(width):
                    self.assertEqual(_integer_expression(loads[1].args[1], dict(
                        self.values, safe_req=safe_req, width=width, safe_logical=safe_logical)),
                        (safe_req * width + safe_logical) * 4)

    def test_every_full_block_pass_waits_and_barriers_outside_active_visibility_guards(self):
        full = [node for node in self.kernel.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "full"]
        self.assertEqual(len(full), 1)
        passes = [node for node in full[0].body if isinstance(node, ast.For)
                  and isinstance(node.target, ast.Name) and node.target.id == "p"]
        self.assertEqual(len(passes), 1)
        loop = passes[0]
        self.assertEqual(ast.unparse(loop.iter), "range_constexpr(WMMA_PASSES)")
        self.assertEqual(len(_calls(loop, "_r16_stage_tdm")), 1)
        self.assertEqual(len(_calls(loop, "fx.rocdl.tdm_ops.tensor_wait")), 1)
        self.assertEqual(len(_calls(loop, "fx.gpu.barrier")), 2)
        # Wait/barriers must be direct pass-body statements, never nested in
        # the SGPR issue guard, per-wave visibility or consumer active guards.
        expressions = [(index, node.value) for index, node in enumerate(loop.body)
                       if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
        stage_guards = [(index, node) for index, node in enumerate(loop.body)
                        if isinstance(node, ast.If) and ast.unparse(node.test) == "stage_active"]
        self.assertEqual(len(stage_guards), 1)
        stage_calls = _calls(stage_guards[0][1], "_r16_stage_tdm")
        self.assertEqual(len(stage_calls), 1)
        self.assertFalse(_calls(stage_guards[0][1], "fx.rocdl.tdm_ops.tensor_wait"))
        self.assertFalse(_calls(stage_guards[0][1], "fx.gpu.barrier"))
        stage = [(stage_guards[0][0], stage_calls[0])]
        wait = [(i, call) for i, call in expressions if ast.unparse(call.func) == "fx.rocdl.tdm_ops.tensor_wait"]
        barriers = [(i, call) for i, call in expressions if ast.unparse(call.func) == "fx.gpu.barrier"]
        self.assertEqual((len(stage), len(wait), len(barriers)), (1, 1, 2))
        self.assertEqual(ast.unparse(wait[0][1]), "fx.rocdl.tdm_ops.tensor_wait(0)")
        active = [index for index, node in enumerate(loop.body) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "pp.active"]
        self.assertEqual(len(active), 1)
        self.assertLess(stage[0][0], wait[0][0])
        self.assertLess(wait[0][0], barriers[0][0])
        self.assertLess(barriers[0][0], active[0])
        self.assertLess(active[0], barriers[1][0])
        self.assertEqual(barriers[1][0], len(loop.body) - 1,
                         "post-consumption barrier must precede next-pass overwrites")
        self.assertEqual(ast.unparse(stage[0][1].args[1]), "storage")
        self.assertEqual(ast.unparse(stage[0][1].args[-1]), "p")

    def test_issue_guard_is_scalar_dedup_and_same_owner_visibility_only(self):
        helper = self.helpers["_r16_stage_active"]
        assignments = {target.id: node.value for node in ast.walk(helper)
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        self.assertEqual(ast.unparse(assignments["wave_id"]), "fx.Int32(fx.rocdl.wave_id())")
        self.assertEqual(ast.unparse(assignments["safe_req"]), "req_ok.select(req, 0)")
        self.assertEqual(ast.unparse(assignments["active"]), "active & (other != req)")
        self.assertEqual(ast.unparse(assignments["vis"]), "_imax(_imin(pos + 1, ctx), 0) // PAGE_SIZE")
        self.assertEqual(ast.unparse(assignments["vmax"]),
                         "_imax(vmax, (owner == req).select(vis, 0))")
        returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertEqual(ast.unparse(returns[0].value), "active & (col0 < vmax)")
        self.assertFalse(_calls(helper, "_ld_i32"))
        self.assertFalse(any(isinstance(node, ast.Name) and node.id in
                             ("lane", "kgrp", "pp", "plans", "n_in") for node in ast.walk(helper)))
        loops = {node.target.id: node for node in ast.walk(helper) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name)}
        self.assertEqual(ast.unparse(loops["prior"].iter), "range_constexpr(p)")
        self.assertEqual(ast.unparse(loops["i"].iter), "range_constexpr(WMMA_ROWS_PER_WAVE)")
        initial_active = [node.value for node in helper.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "active"
                                  for target in node.targets)]
        self.assertEqual([ast.unparse(node) for node in initial_active], ["req_ok"])
        self.assertEqual(len(_calls(self.kernel, "_r16_stage_active")), 1)
        binding = [node.value for node in ast.walk(self.kernel) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "stage_active"
                           for target in node.targets)]
        self.assertEqual(len(binding), 1)
        self.assertEqual(ast.unparse(binding[0].func), "_r16_stage_active")
        self.assertEqual(ast.unparse(binding[0].args[-1]), "p")

    def test_full_guard_is_workgroup_uniform_and_partial_blocks_keep_raw_fallback(self):
        assignments = {target.id: node.value for node in self.kernel.body
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        self.assertEqual(ast.unparse(assignments["full"]), "rblk * 16 + 16 <= rows")
        self.assertEqual({node.id for node in ast.walk(assignments["full"])
                          if isinstance(node, ast.Name)}, {"rblk", "rows"})
        partial = [node for node in self.kernel.body if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "full == 0"]
        self.assertEqual(len(partial), 1)
        self.assertFalse(_calls(partial[0], "_r16_stage_tdm"))
        self.assertFalse(_calls(partial[0], "fx.rocdl.tdm_ops.tensor_wait"))
        self.assertFalse(_calls(partial[0], "fx.gpu.barrier"))
        fallback = _calls(partial[0], "_pass_values")
        self.assertEqual(len(fallback), 1)
        self.assertEqual(ast.unparse(fallback[0].args[1]), "kv_rsrc")
        self.assertEqual(len(fallback[0].args), 10)
        full = next(node for node in self.kernel.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "full")
        staged = _calls(full, "_pass_values")
        self.assertEqual(len(staged), 1)
        self.assertEqual(ast.unparse(staged[0].args[1]), "kptr")
        self.assertEqual(len(staged[0].args), 11)
        self.assertIs(staged[0].args[-1].value, True)
        selection = next(node.value for node in self.build.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "selected_kernel"
                                 for target in node.targets))
        self.assertEqual(ast.unparse(selection),
                         "(kernel_tdm_r16 if use_tdm else kernel) if rows_per_block == 16 else kernel_r64 if rows_per_block == 64 else kernel_r32")

    def test_b_prologue_wave_and_metadata_are_scalar_without_changing_raw_metadata(self):
        assignments = {target.id: node.value for node in self.kernel.body
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        self.assertEqual(ast.unparse(assignments["wave"]), "fx.Int32(fx.rocdl.wave_id())")
        self.assertEqual(len(_calls(self.kernel, "_rows_meta_uniform")), 1)
        self.assertFalse(_calls(self.kernel, "_rows_meta"))
        uniform = self.helpers["_rows_meta_uniform"]
        raw = self.helpers["_rows_meta"]
        # ROUND-NATIVE64: the position load goes through the build-time seam
        # (ld_uniform_pos / ld_pos), the other two loads are unchanged.
        self.assertEqual(len(_calls(uniform, "_ld_uniform_i32")), 2)
        self.assertEqual(len(_calls(uniform, "ld_uniform_pos")), 1)
        self.assertEqual(len(_calls(raw, "ld_pos")), 1)
        self.assertFalse(_calls(uniform, "_ld_i32"))
        # Prove the scalar clone has exactly the raw metadata computation and
        # clamps, differing only in the three load helpers; no code execution.
        normalized = ast.parse(ast.unparse(uniform)).body[0]
        for node in ast.walk(normalized):
            if isinstance(node, ast.Name) and node.id == "_ld_uniform_i32":
                node.id = "_ld_i32"
            elif isinstance(node, ast.Name) and node.id == "ld_uniform_pos":
                node.id = "ld_pos"
        self.assertEqual([ast.dump(node) for node in normalized.body],
                         [ast.dump(node) for node in raw.body])
        for name in ("kernel", "kernel_r32"):
            self.assertEqual(len(_calls(self.helpers[name], "_rows_meta")), 1)
            self.assertFalse(_calls(self.helpers[name], "_rows_meta_uniform"))

    def test_staged_layout_flag_is_forwarded_as_trace_time_helper_selection(self):
        tile = self.helpers["_tile"]
        assignments = {target.id: node.value for node in tile.body
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        self.assertEqual(ast.unparse(assignments["base_fn"]), "_pair_base_tdm if staged else _pair_base_raw")
        self.assertEqual(ast.unparse(assignments["frag_fn"]), "_pair_frag_tdm if staged else _frag")
        self.assertEqual(tile.args.args[-1].arg, "staged")
        self.assertIs(tile.args.defaults[-1].value, False)
        values = self.helpers["_pass_values"]
        self.assertEqual(values.args.args[-1].arg, "staged")
        self.assertIs(values.args.defaults[-1].value, False)
        calls = _calls(values, "_tile")
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[-1]), "staged")


# ---------------------------------------------------------------------------
# Round R16: occupancy-neutral forced-TDM r16, 32-column wave-private tiles.
# Independent of the 64-row helpers above: payload DWORDs are copied straight
# from the source with the descriptor's skip rule; references are per-lane
# global-load fragments.  Byte/layout model only, not ISA or GPU evidence.
# ---------------------------------------------------------------------------
def r16_tile_bytes(tile_cols):
    return tile_cols * TDM_K_ROW_BYTES


def stage_r16_cols_pass(source, tables, t2r, rblk, origin, columns, row_visible, p,
                        tile_cols, *, scratch, active_only=True, wave_order=(0, 1, 2, 3)):
    """Mirror of _r16_stage_active + _r16_stage_tdm for one pass, any tile width."""
    num_pages = len(source) // PAGE_BYTES
    tile_bytes = r16_tile_bytes(tile_cols)
    lds_bytes = R16_WAVES * tile_bytes
    assert len(scratch) == lds_bytes and origin % tile_cols == 0
    scratch = bytearray(scratch)
    issued = {}
    for wave in wave_order:
        t0 = rblk * 16 + wave * 4
        req = t2r[t0 + p]
        req_ok = 0 <= req < len(tables)
        safe_req = req if req_ok else 0
        active = req_ok and all(t2r[t0 + prior] != req for prior in range(p))
        vmax = max([row_visible[t0 + i] if t2r[t0 + i] == req else 0 for i in range(4)])
        issue = (active and origin < vmax) if active_only else True
        issued[wave] = issue
        if not issue:
            continue
        table = tables[safe_req]
        starts = []
        for group in range(tile_cols // INDICES_PER_GATHER):
            local = group * 8
            row_ids = []
            for page in range(2):
                logical = (origin + local) // PAGE_ROWS + page
                logical_ok = logical < len(table)
                phys = table[logical if logical_ok else 0]
                valid = req_ok and logical_ok and 0 <= phys < num_pages
                phys = phys if valid else 0
                row_ids += [phys * PAGE_ROWS + row for row in range(PAGE_ROWS)]
            byte_off = wave * tile_bytes + local * TDM_K_ROW_BYTES
            assert byte_off % 128 == 0 and byte_off + 8 * TDM_K_ROW_BYTES <= (wave + 1) * tile_bytes
            starts.append(byte_off)
            for data_dw in range(8 * 64):
                row_id = row_ids[data_dw // 64]
                src = row_id * ROW_BYTES + (data_dw % 64) * 4
                dst = byte_off + (data_dw + (data_dw // 64) * 4) * 4
                assert 0 <= src <= len(source) - 4
                assert wave * tile_bytes <= dst <= (wave + 1) * tile_bytes - 4
                scratch[dst:dst + 4] = source[src:src + 4]
        issued[wave] = tuple(starts)
    return bytes(scratch), issued


def r16_store_plan(t2r, row_visible, rows, columns, tile_cols, num_requests):
    """Mirror of _pass_plan/_lane_cols/_store_offsets/_pass_values masks: count writes."""
    writes = {}
    grid_x = (columns + tile_cols - 1) // tile_cols
    for rblk in range((rows + 15) // 16):
        for wave in range(4):
            t0 = rblk * 16 + wave * 4
            if t0 >= rows:
                continue          # `live` (raw fallback) / `full` both skip it
            meta = []
            for i in range(4):
                row = t0 + i
                safe_row = min(row, rows - 1)
                meta.append((row, row < rows, t2r[safe_row]))
            for chunk in range(grid_x):
                col0 = chunk * tile_cols
                for p in range(4):
                    row_p, ok_p, req_p = meta[p]
                    active = ok_p and all(meta[q][0] >= rows or meta[q][2] != req_p
                                          for q in range(p))
                    if not active:
                        continue
                    for lane in range(32):
                        kgrp, n_in = lane // 16, lane % 16
                        for nt in range(tile_cols // 16):
                            c = col0 + nt * 16 + n_in
                            for i in (2 * kgrp, 2 * kgrp + 1):
                                own = meta[i][1] and meta[i][2] == req_p
                                if own and c < columns:
                                    row = min(t0 + i, rows - 1)
                                    off = min(row * columns + c, rows * columns - 1) * 4
                                    assert 0 <= off < rows * columns * 4
                                    writes[off] = writes.get(off, 0) + 1
    return writes


class KR16Cols32GatherCPU(unittest.TestCase):
    """Byte exactness of the 32-column r16 TDM tile for every lane/k-chunk."""

    TILE_COLS = 32

    @classmethod
    def setUpClass(cls):
        KTileGatherCPU.setUpClass()
        cls.source = KTileGatherCPU.source
        cls.num_pages = KTileGatherCPU.num_pages
        cls.tables = tuple(tuple((request * 13 + logical * 29 + 7) % cls.num_pages
                                 for logical in range(33)) for request in range(8))

    def lds(self):
        return r16_tile_bytes(self.TILE_COLS) * R16_WAVES

    def check_pass(self, tables, t2r, p, origin, columns, row_visible, scratch, rblk=0,
                   active_only=True, wave_order=(0, 1, 2, 3)):
        tile_cols = self.TILE_COLS
        tile_bytes = r16_tile_bytes(tile_cols)
        actual, issued = stage_r16_cols_pass(self.source, tables, t2r, rblk, origin, columns,
                                             row_visible, p, tile_cols, scratch=scratch,
                                             active_only=active_only, wave_order=wave_order)
        self.assertEqual(len(actual), self.lds())
        for wave in range(4):
            base = wave * tile_bytes
            t0 = rblk * 16 + wave * 4
            req = t2r[t0 + p]
            req_ok = 0 <= req < len(tables)
            if issued[wave] is False:
                self.assertEqual(actual[base:base + tile_bytes], scratch[base:base + tile_bytes])
                continue
            self.assertEqual(issued[wave], tuple(base + g * 2176 for g in range(tile_cols // 8)))
            table = tables[req if req_ok else 0]
            for nt in range(tile_cols // 16):
                for lane in range(32):
                    c = origin + nt * 16 + lane % 16
                    logical_ok = c // 4 < len(table)
                    phys = table[c // 4 if logical_ok else 0]
                    page_ok = req_ok and logical_ok and 0 <= phys < self.num_pages
                    phys = phys if page_ok else 0
                    for s in range(4):
                        g = phys * PAGE_BYTES + (c % 4) * ROW_BYTES + (lane // 16) * 16 + s * 64
                        expected = self.source[g:g + 16] + self.source[g + 32:g + 48]
                        # Kernel consumer: (c % tdm_cols) * 272 + kgrp * 16 + s * 64 (+32).
                        lo = base + (c % tile_cols) * 272 + (lane // 16) * 16 + s * 64
                        hi = lo + 32
                        for off in (lo, hi):
                            self.assertEqual(off % 16, 0)
                            self.assertGreaterEqual(off, base)
                            self.assertLessEqual(off + 16, base + tile_bytes)
                            self.assertLessEqual((off - base) % 272 + 16, 256,
                                                 "consumer vector touches LDS padding")
                        self.assertEqual(actual[lo:lo + 16] + actual[hi:hi + 16], expected,
                                         f"wave={wave} p={p} nt={nt} s={s} lane={lane}")
            for row in range(tile_cols):
                pad = base + row * 272 + 256
                self.assertEqual(actual[pad:pad + 16], scratch[pad:pad + 16], "padding written")
        return actual, issued

    def test_lds_footprint_alignment_and_occupancy_ratio(self):
        self.assertEqual(r16_tile_bytes(32), 8704)
        self.assertEqual(self.lds(), 34816)
        self.assertEqual(R16_LDS_BYTES, 69632)
        self.assertEqual(R16_LDS_BYTES // self.lds(), 2)
        self.assertEqual(r16_tile_bytes(32) % 128, 0)
        # Per pass a wave issues 4 eight-row ops (32 rows), 16 over four passes:
        # far below the 63-op completion counter; XACK 3/wave is HW throttled.
        self.assertEqual(self.TILE_COLS // INDICES_PER_GATHER, 4)
        self.assertLessEqual(R16_PASSES * self.TILE_COLS // INDICES_PER_GATHER, 63)

    def test_every_lane_kchunk_all_passes_mixed_requests_and_origins(self):
        t2r = (0, 1, 2, 3, 4, 3, 2, 1, 7, 5, 4, 6, 2, 6, 0, 5)
        for origin in (0, 32, 64, 96, 128):
            scratch = PADDING_SENTINEL * (self.lds() // 16)
            for p in range(4):
                with self.subTest(origin=origin, p=p):
                    scratch, issued = self.check_pass(self.tables, t2r, p, origin, 132,
                                                      [132] * 16, scratch, active_only=False)
                    payloads = [b"".join(scratch[w * 8704 + r * 272:w * 8704 + r * 272 + 256]
                                         for r in range(32)) for w in range(4)]
                    self.assertEqual(len(set(payloads)), len({t2r[w * 4 + p] for w in range(4)}))

    def test_request_dedup_visibility_and_invalid_requests_skip_issue(self):
        t2r = (0, 0, 1, 1, 2, 3, 2, 4, -1, 0, 8, 1, 5, 6, 7, 5)
        visible = (65, 65, 132, 132, 32, 0, 32, 132, 132, 132, 132, 132, 132, 0, 132, 132)
        for origin in (0, 32, 64):
            scratch = bytes([0xD3]) * self.lds()
            for p in range(4):
                with self.subTest(origin=origin, p=p):
                    scratch, issued = self.check_pass(self.tables, t2r, p, origin, 132,
                                                      visible, scratch)
                    for wave in range(4):
                        req = t2r[wave * 4 + p]
                        vis = max(visible[wave * 4 + i] for i in range(4)
                                  if t2r[wave * 4 + i] == req)
                        expect = (0 <= req < 8 and req not in t2r[wave * 4:wave * 4 + p]
                                  and origin < vis)
                        self.assertEqual(issued[wave] is not False, expect)

    def test_invalid_pages_and_logical_tails_select_physical_page_zero(self):
        tables = tuple(tuple((7, -1, self.num_pages, 0x7FFFFFFF, 3)[(i + req) % 5]
                             for i in range(9)) for req in range(3))
        t2r = (-1, 0, 2, 1, 2, -0x80000000, 1, 0x7FFFFFFF, 0, 1, 2, 2, 1, -1, 1, 2)
        for origin in (0, 32):      # width 9 pages = 36 columns: origin 32 has a tail
            scratch = PADDING_SENTINEL * (self.lds() // 16)
            for p in range(4):
                with self.subTest(origin=origin, p=p):
                    scratch, _ = self.check_pass(tables, t2r, p, origin, 33, [36] * 16,
                                                 scratch, active_only=False)

    def test_padding_canaries_survive_and_wave_orders_commute(self):
        rng = random.Random(0x34816)
        canaries = bytes(rng.randrange(256) for _ in range(self.lds()))
        t2r = tuple((row * 3 + row // 4) % 8 for row in range(48))
        for rblk in (0, 1, 2):
            scratch = canaries
            for p in range(4):
                results = set()
                for order in ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2)):
                    actual, _ = self.check_pass(self.tables, t2r, p, 64, 132, [132] * 48,
                                                scratch, rblk=rblk, active_only=False,
                                                wave_order=order)
                    results.add(actual)
                self.assertEqual(len(results), 1)
                scratch = results.pop()

    def test_logical_64_bank_spread_matches_c64_layout(self):
        for nt in range(2):
            for s in range(4):
                for half in range(2):
                    banks = []
                    for lane in range(half * 16, half * 16 + 16):
                        lo = (nt * 16 + lane % 16) * 272 + (lane // 16) * 16 + s * 64
                        banks += [(lo // 4 + d) % 64 for d in range(4)]
                    self.assertEqual(sorted(banks), list(range(64)))

    def test_every_logit_written_exactly_once_with_32_column_blocks(self):
        rng = random.Random(0x3232)
        cases = [(64, 16, 65536 // 64), (16, 16, 33), (17, 3, 1), (31, 2, 255),
                 (64, 16, 1000), (48, 5, 257), (7, 9, 31), (64, 4, 64)]
        for rows, requests, columns in cases:
            t2r = [rng.choice([-1, requests] + list(range(requests))) for _ in range(rows)]
            for tile_cols in (32, 64):
                with self.subTest(rows=rows, requests=requests, columns=columns, cols=tile_cols):
                    writes = r16_store_plan(t2r, [columns] * rows, rows, columns,
                                            tile_cols, requests)
                    self.assertEqual(len(writes), rows * columns)
                    self.assertEqual(set(writes.values()), {1})
                    self.assertEqual(set(writes), {4 * i for i in range(rows * columns)})


class KR16Cols32KernelSourceCPU(unittest.TestCase):
    """Source bindings specific to the 32-column r16 TDM kernel."""

    @classmethod
    def setUpClass(cls):
        KR16PerWaveKernelSourceCPU.setUpClass()
        cls.helpers = KR16PerWaveKernelSourceCPU.helpers
        cls.kernel = KR16PerWaveKernelSourceCPU.kernel
        cls.tree = KR16PerWaveKernelSourceCPU.tree

    def test_store_loops_follow_actual_tile_count(self):
        loops = [node for node in ast.walk(self.kernel) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id.startswith("j")]
        self.assertEqual(len(loops), 4)
        self.assertEqual(sorted(ast.unparse(loop.iter) for loop in loops),
                         ["range_constexpr(len(msk))", "range_constexpr(len(msk))",
                          "range_constexpr(len(res))", "range_constexpr(len(res))"])
        for name in ("_pass_values", "_pass_masks"):
            loop = next(node for node in ast.walk(self.helpers[name]) if isinstance(node, ast.For))
            self.assertEqual(ast.unparse(loop.iter), "range_constexpr(len(cols))")
        lane_cols = self.helpers["_lane_cols"]
        self.assertEqual(ast.unparse(lane_cols.args.defaults[-1]), "WMMA_N_TILES")
        for name in ("kernel", "kernel_r32"):
            calls = _calls(self.helpers[name], "_lane_cols")
            self.assertEqual([ast.unparse(c) for c in calls], ["_lane_cols(col0, n_in)"])
            col0 = [node.value for node in self.helpers[name].body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "col0" for t in node.targets)]
            self.assertEqual([ast.unparse(v) for v in col0], ["chunk * WMMA_COLS"])

    def test_environment_switch_and_host_grid_binding(self):
        consts = {t.id: node.value for node in self.tree.body if isinstance(node, ast.Assign)
                  for t in node.targets if isinstance(t, ast.Name)}
        self.assertIn("'QSA_BF16_R16_TDM_COLS', '32'", ast.unparse(consts["_BF16_R16_TDM_COLS"]))
        wrapper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "qsa_paged_mqa_logits_page4")
        text = ast.unparse(wrapper)
        # ROUND-NATIVE64: int32 callers expand **qp_kw to no keyword at all.
        self.assertIn("launch = _build_wmma_kernel(float(divisor), rows_per_block, k_load, **qp_kw)", text)
        self.assertIn("wmma_grid(rows, columns, rows_per_block, launch.cols_per_block)", text)


if __name__ == "__main__":
    unittest.main()

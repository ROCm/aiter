# SPDX-License-Identifier: MIT
"""ROUND-NATIVE64 CPU gate: int64 query_positions read natively by the kernel.

No GPU, no compiler. The real kernel source is executed through the
``trace_check_cpu`` mock (wave32 torch lanes, strict i32 overflow checks, plane
extent/OOB checks on every load).

1. Seam semantics on 36 boundary values (+ dense neighbourhoods and random
   int64s): the real ``_ld_pos_i64`` (per-lane 8-byte load) and the real
   ``_ld_uniform_pos_i64`` (SGPR path, two dword loads; the LLVM intrinsic is
   replaced by a plane reader that checks offsets) return exactly the golden
   clamp ``max(INT32_MIN, min(INT32_MAX-1, v))`` == ROUND-NOSYNC's wrapper
   ``_to_i32_saturated(v, INT32_MAX-1)``; and the resulting visible-group
   arithmetic equals the golden int64 formula for every (pos, ctx) boundary
   pair. ``_ld_pos_i32``/``_ld_uniform_pos_i32`` are ``row * 4`` dword loads.
2. Whole-kernel trace (bf16 WMMA r16 and r32, raw K): ``pos64=True`` on the
   raw int64 positions is *bitwise identical* to the int32 kernel run on the
   wrapper-narrowed positions, and matches ``qsa_golden.reference_logits`` on
   int64 boundary positions (INT32_MAX, 2**31, 2**32+3, +-2**40, int64
   extremes, ...). Descriptor extent of the position plane is rows * 8.
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("QSA_BF16_K_LOAD", "raw")    # trace mock has no TDM
import trace_check_cpu as T  # noqa: E402

K = T.load_module()
G, S = T.G, T.S

I32_MIN, I32_MAX = -(2**31), 2**31 - 1
I64_MIN, I64_MAX = -(2**63), 2**63 - 1
BOUNDARY = sorted({
    0, 1, -1, 2, 3, 4, 5, 7, 8, 4095, 2**29 - 1, 2**29, 2**29 + 1,
    I32_MAX - 4, I32_MAX - 3, I32_MAX - 2, I32_MAX - 1, I32_MAX, 2**31, 2**31 + 1,
    2**32 - 1, 2**32, 2**32 + 1, 2**32 + 5, 2**33 + 3, 2**40, 2**62, I64_MAX,
    I32_MIN, I32_MIN + 1, I32_MIN - 1, -(2**32), -(2**32) + 5, -(2**40), -(2**62), I64_MIN,
})
assert len(BOUNDARY) == 36


def golden_clamp(v):
    return max(I32_MIN, min(I32_MAX - 1, v))


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - 2**32 if value >= 2**31 else value


def kernel_visible(req_ok, pos_i32, ctx_i32):
    if not req_ok:
        return 0
    return max(min(wrap32(pos_i32 + 1), ctx_i32), 0) // 4


def golden_visible(pos, ctx):
    return max(0, min((pos + 1) // 4, ctx // 4))


def _values():
    vals = list(BOUNDARY)
    for centre in (0, I32_MAX, I32_MIN, 2**32, -(2**32), 2**31, I64_MAX - 8, I64_MIN + 8):
        vals += [centre + d for d in range(-8, 9) if I64_MIN <= centre + d <= I64_MAX]
    rng = random.Random(0x64)
    vals += [rng.randint(I64_MIN, I64_MAX) for _ in range(512)]
    vals += [rng.randint(-(2**34), 2**34) for _ in range(512)]
    return vals


def _seam_rows(loader, values, item):
    """Run ``loader(rsrc, row)`` over every row, 32 lanes at a time."""
    src = torch.tensor(values, dtype=torch.int64)
    plane = src.view(torch.int32) if item == 8 else src.to(torch.int32)
    T.Sim.names = {id(plane): "qp"}
    rsrc = K._plane(plane, K._plane_bytes(T.Int32(len(values)), item))
    assert T.Sim.planes["qp"].extent == len(values) * item
    out = []
    for base in range(0, len(values), T.L):
        rows = torch.clamp(torch.arange(T.L) + base, max=len(values) - 1)
        T.Sim.mask = [torch.ones(T.L, dtype=torch.bool)]
        got = loader(rsrc, T.MV(rows.to(torch.int64), T.Int32))
        out += got.t.tolist()[: min(T.L, len(values) - base)]
    return out


class SeamSemanticsCPU(unittest.TestCase):
    def test_loader_table(self):
        self.assertEqual(K._pos_loaders(False)[2:], (4, ""))
        self.assertEqual(K._pos_loaders(True)[2:], (8, "_qp64"))
        self.assertIs(K._pos_loaders(False)[0], K._ld_pos_i32)
        self.assertIs(K._pos_loaders(True)[1], K._ld_uniform_pos_i64)

    def test_lane_loader_matches_golden_clamp_and_nosync_wrapper(self):
        values = _values()
        got = _seam_rows(K._ld_pos_i64, values, 8)
        want = [golden_clamp(v) for v in values]
        self.assertEqual(got, want)
        nosync = K._to_i32_saturated(torch.tensor(values, dtype=torch.int64), I32_MAX - 1)
        self.assertEqual(got, nosync.tolist())
        for v, g in zip(BOUNDARY, _seam_rows(K._ld_pos_i64, BOUNDARY, 8)):
            with self.subTest(v=v):
                self.assertEqual(g, golden_clamp(v))

    def test_uniform_loader_offsets_and_values(self):
        offsets = []

        def ld_uniform(rsrc, byte_off):
            offs = byte_off.t if isinstance(byte_off, T.MV) else torch.full((T.L,), int(byte_off))
            offsets.append(offs.clone())
            # same plane checks as a buffer load (in bounds, dword aligned)
            return T._raw_load(T.Int32, rsrc, T.MV(offs, T.Int32), None)

        saved = K._ld_uniform_i32
        K._ld_uniform_i32 = ld_uniform
        try:
            values = _values()
            got = _seam_rows(K._ld_uniform_pos_i64, values, 8)
        finally:
            K._ld_uniform_i32 = saved
        self.assertEqual(got, [golden_clamp(v) for v in values])
        # exactly two dword loads per row: lo at row*8, hi at row*8+4
        self.assertTrue(all(bool((b - a == 4).all()) for a, b in zip(offsets[0::2], offsets[1::2])))
        self.assertTrue(all(bool((a % 8 == 0).all()) for a in offsets[0::2]))

    def test_int32_seam_is_plain_dword_load(self):
        values = [v for v in BOUNDARY if I32_MIN <= v <= I32_MAX]
        self.assertEqual(_seam_rows(K._ld_pos_i32, values, 4), values)  # no clamp at all

    def test_visible_groups_all_boundary_pairs(self):
        pos_i32 = dict(zip(BOUNDARY, _seam_rows(K._ld_pos_i64, BOUNDARY, 8)))
        max_column = (I32_MAX // 4) - 1
        for pos in BOUNDARY:
            for ctx in BOUNDARY:
                # contexts reach the kernel via the (unchanged) NOSYNC clamp
                ctx_i32 = max(I32_MIN, min(I32_MAX, ctx))
                gold = golden_visible(pos, ctx)
                got = kernel_visible(True, pos_i32[pos], ctx_i32)
                with self.subTest(pos=pos, ctx=ctx):
                    self.assertEqual(got, gold if gold < 2**29 else 2**29 - 1)
                    for col in {0, 1, max(gold - 1, 0), min(gold, max_column),
                                min(got, max_column), max(got - 1, 0), max_column}:
                        if col <= max_column:      # every addressable column
                            self.assertEqual(col < got, col < gold, col)
                self.assertEqual(kernel_visible(False, pos_i32[pos], ctx_i32), 0)


def _run(name, regime, rows, requests, width, pages, positions, rows_per_block):
    os.environ["QSA_TRACE_ROWS_PER_BLOCK"] = str(rows_per_block)
    case = S.make_case("native64", regime, rows=rows, requests=requests, max_ctx=16 * width,
                       index_dtype="bf16", table_width=width, cache_pages=pages)
    data = G.build_case(case, device="cpu", seed=3)
    t2r = data["token_to_request"].clone()
    ctx = data["context_lens"].clone()
    table = data["page_table"].clone()
    qpos = data["query_positions"].clone().to(torch.int64)
    for i, v in enumerate(positions[:rows]):
        qpos[i] = v
    cols = case.columns
    T.Sim.n_loads = 0
    out64, vis64, rst64 = T.run_kernel(K, data["q"], data["compressed_k_cache"], table, t2r,
                                       qpos, ctx, cols, pos64=True)
    narrowed = K._to_i32_saturated(qpos, I32_MAX - 1)
    out32, vis32, rst32 = T.run_kernel(K, data["q"], data["compressed_k_cache"], table, t2r,
                                       narrowed, ctx, cols)
    gold, gold_vis, valid = G.reference_logits(
        data["q"], data["compressed_k_cache"], table, t2r, qpos, ctx,
        compress_ratio=S.COMPRESS_RATIO, max_columns=cols)
    res = G.compare_logits(out64, vis64, gold, gold_vis, valid)
    return dict(bitwise=bool(torch.equal(out64.view(torch.int32), out32.view(torch.int32))
                             and torch.equal(vis64, vis32) and torch.equal(rst64, rst32)),
                golden=bool(res.get("pass")), checked=res.get("checked"),
                vis=vis64.tolist(), gold_vis=gold_vis.tolist())


# I64_MAX itself is excluded here only because the golden's own int64 `pos + 1`
# wraps there (torch); the seam tests above cover I64_MAX with exact math.
EDGE = [I32_MAX, 2**31, 2**32 + 3, 2**40, -(2**40), I64_MAX - 1, I64_MIN, -1,
        I32_MAX - 1, I32_MIN, 2**32 - 1, 2**33 + 3, 0, 5, I32_MIN - 1, 2**62]


class KernelTraceCPU(unittest.TestCase):
    def check(self, *args):
        r = _run(*args)
        self.assertTrue(r["bitwise"], r)
        self.assertTrue(r["golden"], r)
        self.assertEqual(r["vis"], [int(v) for v in r["gold_vis"]])
        return r

    def test_r16_decode_and_draft(self):
        for rows_per_block in (16, 32):
            with self.subTest(r=rows_per_block):
                self.check("decode", "decode", 16, 4, 16, 64, EDGE, rows_per_block)
                self.check("draft", "draft", 16, 16, 16, 512, EDGE[::-1], rows_per_block)

    def test_r32_prefill_full_and_tail(self):
        # 64 rows of one request (r32 shared-B path) and a 13-row tail.
        self.check("prefill64", "prefill", 64, 1, 16, 64, EDGE * 4, 32)
        self.check("prefill13", "prefill", 13, 2, 32, 128, EDGE, 32)


if __name__ == "__main__":
    unittest.main()

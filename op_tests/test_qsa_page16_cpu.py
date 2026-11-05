# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: S102, RUF012 -- exec of kernel source fragments for CPU lane emulation; read-only class tables
"""ROUND-PAGE16 CPU gates: index page P in {4, 16} (KV block 16 / 64). No GPU, no FlyDSL.

1. Lane-level emulation of the kernel's K address math. The *real* trace-time
   helpers of the kernel source (TDM gather producers ``_r16_stage_tdm``,
   ``_pair_stage_tdm`` (bf16 + fp8), ``_persist_issue_tdm``, draft ``_stage_tdm``,
   decode-group ``_issue``, the module helper ``_p16_row_ids``, and the consumer
   address helpers ``_pair_base_tdm`` / ``_pair_base_raw`` / ``_kbase_tdm`` /
   ``_kbase_raw``) are extracted from the source with ``ast`` and executed against
   a fake ``fx`` (Python ints with i32 select/compare semantics, bounds-checked
   ``s_buffer_load``) and a recording TDM gather (LDS byte offset -> source row,
   bounds-checked against the descriptor's ``num_pages * P`` rows). Every
   (row, column) a kernel variant stores is then resolved to the K row its WMMA B
   fragment reads, and the resulting logits are compared with the frozen golden
   ``qsa_golden.reference_logits`` / ``compare_logits`` on ``qsa_golden.build_case``
   fixtures (unmodified, page_size=16) with edge cases (invalid / out-of-range
   pages, invalid requests, tails, visible < columns, narrow page tables).
2. P = 4 branches of every TDM site are textually identical to the base kernel
   (bf492262, branch base a2c4cb3b) and the builders pass no new keyword at P = 4.
3. Op/profile: page-16 rules (schema, matcher, page isolation, reasons, prewarm
   geometry routes) using the standalone package loader (no aiter import).

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_page16_cpu.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import types
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KREL = "aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py"
KERNEL = os.path.join(ROOT, *KREL.split("/"))
PKG_DIR = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
BASE_REV = os.environ.get("QSA_PAGE16_BASE_REV", "a2c4cb3b")
BASE_SHA = "bf49226206b999f948e8b1eb0605935157b3d291b0a0b243581897d8ca661b93"


def _golden_dir() -> str:
    for c in (
        os.environ.get("QSA_GOLDEN_DIR", ""),
        os.path.join(ROOT, "..", "..", "..", "..", "golden_baseline"),
        "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline",
    ):
        if c and os.path.isfile(os.path.join(c, "qsa_golden.py")):
            return os.path.abspath(c)
    raise RuntimeError("frozen golden not found: set QSA_GOLDEN_DIR")


sys.path.insert(0, _golden_dir())
import qsa_golden as G
import qsa_shapes as S

with open(KERNEL, encoding="utf-8") as _fh:
    SRC = _fh.read()
TREE = ast.parse(SRC)


# ---------------------------------------------------------------------------
# Fake fx: i32 scalars (SGPR/VGPR values of one lane) with the kernel's ops
# ---------------------------------------------------------------------------
I32_MIN, I32_MAX = -(2**31), 2**31 - 1


class FB:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = bool(v.v if isinstance(v, FB) else v)

    def __and__(self, o):
        return FB(self.v and _b(o))

    __rand__ = __and__

    def __or__(self, o):
        return FB(self.v or _b(o))

    def __eq__(self, o):
        return FB(self.v == _b(o))

    def __bool__(self):
        raise AssertionError("trace-time bool() of a runtime predicate")

    def select(self, a, b):
        return FI(a) if self.v else FI(b)


def _b(x):
    return x.v if isinstance(x, FB) else bool(x)


def _i(x):
    if isinstance(x, FI):
        return x.v
    assert isinstance(x, int) and not isinstance(x, bool), x
    return x


class FI:
    __slots__ = ("v",)

    def __init__(self, v):
        v = _i(v)
        assert I32_MIN <= v <= I32_MAX, f"i32 overflow {v}"
        self.v = v

    def __add__(self, o):
        return FI(self.v + _i(o))

    __radd__ = __add__

    def __sub__(self, o):
        return FI(self.v - _i(o))

    def __rsub__(self, o):
        return FI(_i(o) - self.v)

    def __mul__(self, o):
        return FI(self.v * _i(o))

    __rmul__ = __mul__

    def __floordiv__(self, o):
        o = _i(o)
        assert self.v >= 0 and o > 0, "signed i32 division on a negative operand"
        return FI(self.v // o)

    def __mod__(self, o):
        o = _i(o)
        assert self.v >= 0 and o > 0
        return FI(self.v % o)

    def __lt__(self, o):
        return FB(self.v < _i(o))

    def __le__(self, o):
        return FB(self.v <= _i(o))

    def __gt__(self, o):
        return FB(self.v > _i(o))

    def __ge__(self, o):
        return FB(self.v >= _i(o))

    def __eq__(self, o):
        return FB(self.v == _i(o))

    def __ne__(self, o):
        return FB(self.v != _i(o))

    __hash__ = None


class Plane:
    """A buffer descriptor: int32 words, bounds-checked like a clamped kernel load."""

    def __init__(self, words, name="pt"):
        self.words = np.asarray(words, dtype=np.int64).reshape(-1)
        self.name = name

    def load(self, byte_off):
        off = _i(byte_off)
        assert off % 4 == 0 and 0 <= off < 4 * self.words.size, f"load OOB {off}"
        return FI(int(self.words[off // 4]))


class Env:
    """Fake fx + recording TDM for one emulated wave."""

    def __init__(self):
        self.wave = 0
        self.lds: dict[int, int] = {}  # LDS byte offset of a staged row -> source row
        self.gathers = 0
        self.lookups = 0
        rocdl = types.SimpleNamespace(
            wave_id=lambda: self.wave,
            tdm_ops=types.SimpleNamespace(tensor_wait=lambda n: None),
            s_wait_dscnt=lambda n: None,
        )
        self.fx = types.SimpleNamespace(
            Int32=FI, rocdl=rocdl, gpu=types.SimpleNamespace(barrier=lambda: None)
        )

    def ld_uniform(self, rsrc, byte_off):
        self.lookups += rsrc.name == "pt"  # page-table s_buffer_loads only
        return rsrc.load(byte_off)

    def gather(self, pitch):
        def _g(KV, ptr, row_ids, num_pages, byte_off, page_size=4):
            assert len(row_ids) == 8, "32-bit index TDM gathers move 8 rows"
            self.gathers += 1
            base = _i(byte_off)
            for k, rid in enumerate(row_ids):
                rid = _i(rid)
                assert (
                    0 <= rid < _i(num_pages) * page_size
                ), f"row {rid} outside the tensor"
                self.lds[base + k * pitch] = rid

        return _g


def _builder(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def _nested(builder, name):
    for node in ast.walk(_builder(builder)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(SRC, node))
    raise KeyError(f"{builder}.{name}")


def _top(name):
    return ast.get_source_segment(SRC, _builder(name))


def _const(name):
    for node in TREE.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == name
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


C = {
    k: _const(k)
    for k in (
        "PAGE_SIZE",
        "COMPRESS_RATIO",
        "SUPPORTED_PAGE_SIZES",
        "HEAD_DIM",
        "TDM_PAD_INTERVAL_DWORDS",
        "TDM_PAD_AMOUNT_DWORDS",
        "WMMA_ROWS_PER_WAVE",
        "TDM_FP8_PAD_INTERVAL_DWORDS",
        "TDM_FP8_PAD_AMOUNT_DWORDS",
    )
}
ROW = (C["TDM_PAD_INTERVAL_DWORDS"] + C["TDM_PAD_AMOUNT_DWORDS"]) * 4  # 272
ROW8 = (C["TDM_FP8_PAD_INTERVAL_DWORDS"] + C["TDM_FP8_PAD_AMOUNT_DWORDS"]) * 4  # 144


def _exec(env: Env, P: int, src: str, name: str, **closure):
    ns = {
        "fx": env.fx,
        "range_constexpr": range,
        "_ld_uniform_i32": env.ld_uniform,
        "_imin": lambda a, b: FI(min(_i(a), _i(b))),
        "_imax": lambda a, b: FI(max(_i(a), _i(b))),
        "P": P,
        "COMPRESS_RATIO": C["COMPRESS_RATIO"],
        "TDM_K_ROW_BYTES": ROW,
        "TDM_FP8_K_ROW_BYTES": ROW8,
        "TDM_K_TILE_BYTES": 64 * ROW,
        "WMMA_ROWS_PER_WAVE": C["WMMA_ROWS_PER_WAVE"],
        "SUPPORTED_PAGE_SIZES": C["SUPPORTED_PAGE_SIZES"],
    }
    exec(_top("_check_page_size"), ns)
    exec(_top("_p16_row_ids"), ns)
    ns.update(closure)
    exec(src, ns)
    return ns[name]


# ---------------------------------------------------------------------------
# Variant emulators: (row, column) -> K source row read by the WMMA B operand
# ---------------------------------------------------------------------------
def _meta(fx_):
    t2r = fx_["token_to_request"].cpu().numpy().astype(np.int64)
    pos = fx_["query_positions"].cpu().numpy().astype(np.int64)
    ctx = fx_["context_lens"].cpu().numpy().astype(np.int64)
    req_n = ctx.size
    ok = (t2r >= 0) & (t2r < req_n)
    sreq = np.where(ok, t2r, 0)
    vis = np.where(ok, np.maximum(np.minimum(pos + 1, ctx[sreq]), 0) // 4, 0)
    return t2r, ok, sreq, vis


def _consumer_page(P, pt, req, req_ok, c, width, pages):
    logical = c // P
    if not req_ok or logical >= width:
        return False, 0
    phys = int(pt[req, logical])
    return (0 <= phys < pages), phys


class Emu:
    def __init__(self, fx_, P, variant, columns):
        self.fx_, self.P, self.variant, self.columns = fx_, P, variant, columns
        self.pt = fx_["page_table"].cpu().numpy().astype(np.int64)
        self.pages = int(fx_["compressed_k_cache"].shape[0])
        self.width = int(self.pt.shape[1])
        self.ptp = Plane(self.pt)
        self.t2r, self.ok, self.sreq, self.vis = _meta(fx_)
        self.t2rp = Plane(self.t2r, "t2r")
        self.nreq = int(fx_["context_lens"].numel())
        self._fns = {}
        self.rows = self.t2r.size
        self.stats = {"gathers": 0, "lookups": 0}

    # producers -------------------------------------------------------------
    def _stage(self, kind, req, col0, wave, rblk=0, p=0, buf=0):
        env, P = Env(), self.P
        env.wave = wave
        width, pages, nreq = FI(self.width), FI(self.pages), FI(self.nreq)
        if kind == "r16":
            fn = _exec(
                env,
                P,
                _nested("_build_wmma_kernel", "_r16_stage_tdm"),
                "_r16_stage_tdm",
                _tdm_k_gather=env.gather(ROW),
                tdm_cols=32,
                tdm_tile_bytes=32 * ROW,
            )
            fn(
                None,
                None,
                self.ptp,
                self.t2rp,
                FI(rblk),
                FI(col0),
                width,
                pages,
                nreq,
                p,
            )
            base = wave * 32 * ROW
        elif kind in ("r32", "fp8r32"):
            b = "_build_wmma_kernel" if kind == "r32" else "_build_wmma_fp8_kernel"
            pitch = ROW if kind == "r32" else ROW8
            fn = _exec(
                env,
                P,
                _nested(b, "_pair_stage_tdm"),
                "_pair_stage_tdm",
                _tdm_k_gather=env.gather(ROW),
                _tdm_fp8_k_gather=env.gather(ROW8),
            )
            fn(None, None, self.ptp, self.t2rp, FI(rblk), FI(col0), width, pages)
            base = 0
            self._pitch = pitch
        elif kind == "persist":
            fn = _exec(
                env,
                P,
                _nested("_build_wmma_kernel", "_persist_issue_tdm"),
                "_persist_issue_tdm",
                _tdm_k_gather=env.gather(ROW),
            )
            fn(None, None, self.ptp, FI(req), FI(col0), width, pages, FI(buf))
            base = buf * 64 * ROW
        elif kind == "draft":
            fn = _exec(
                env,
                P,
                _nested("_build_draft_kernel", "_stage_tdm"),
                "_stage_tdm",
                gather=env.gather(ROW),
                wave_cols=32,
                tile_bytes=32 * ROW,
                pitch=ROW,
            )
            ok = FB(0 <= req < self.nreq)
            fn(
                None,
                None,
                self.ptp,
                FI(req if ok.v else 0),
                ok,
                FI(col0),
                width,
                pages,
                FI(wave),
            )
            base = wave * 32 * ROW
        elif kind == "decg":
            fn = _exec(
                env,
                P,
                _nested("_build_decode_group_kernel", "_issue"),
                "_issue",
                gather=env.gather(ROW),
                wave_cols=32,
                tile_bytes=32 * ROW,
                pitch=ROW,
            )
            ok = FB(0 <= req < self.nreq)
            pp = types.SimpleNamespace(safe_req=FI(req if ok.v else 0), req_ok=ok)
            fn(None, None, self.ptp, pp, FI(col0), width, pages, FI(wave))
            base = wave * 32 * ROW
        else:
            raise ValueError(kind)
        self.stats["gathers"] += env.gathers
        self.stats["lookups"] += env.lookups
        return env.lds, base

    # consumer address helpers (real source) ---------------------------------
    def _fn(self, key):
        f = self._fns.get(key)
        if f is None:
            env, P = Env(), self.P
            if key == "draft":
                f = _exec(
                    env,
                    P,
                    _nested("_build_draft_kernel", "_kbase_tdm"),
                    "_kbase_tdm",
                    pitch=ROW,
                    is_fp8=False,
                )
            elif key == "raw":
                f = _exec(
                    env,
                    P,
                    _nested("_build_wmma_kernel", "_pair_base_raw"),
                    "_pair_base_raw",
                    page_bytes=P * 256,
                    key_row_bytes=256,
                )
            else:
                f = _exec(
                    env,
                    P,
                    _nested("_build_wmma_kernel", "_pair_base_tdm"),
                    "_pair_base_tdm",
                    tdm_cols={"r16": 32}.get(key, 64),
                )
            self._fns[key] = f
        return f

    def _tdm_row(self, lds, base, local, pitch, kind):
        if kind == "draft":
            off = _i(self._fn("draft")(None, None, FI(local), 0))
        elif (
            kind == "decg"
        ):  # decode-group inlines `k_base = local * pitch + kgrp * kgrp_bytes`
            off = local * pitch
        else:  # c % tdm_cols with c == local (col0 = 0)
            off = _i(
                self._fn(kind if kind == "r16" else "r32")(None, None, FI(local), 0)
            )
        assert off % pitch == 0
        return lds[base + off]

    def raw_row(self, phys, off):
        addr = _i(self._fn("raw")(FI(phys), FI(off), None, 0))
        assert addr % 256 == 0
        return addr // 256

    # full emulation ----------------------------------------------------------
    def run(self):
        """K source row per (row, column) or -1 (masked: -inf)."""
        P, cols, v = self.P, self.columns, self.variant
        src = np.full((self.rows, cols), -1, dtype=np.int64)
        span = {
            "r16": 32,
            "r32": 64,
            "fp8r32": 64,
            "persist": 512,
            "draft": 128,
            "decg": 128,
            "raw": 64,
        }[v]
        cache = {}
        for r in range(self.rows):
            if not self.ok[r]:
                continue
            req = int(self.sreq[r])
            for c in range(min(cols, int(self.vis[r]))):
                page_ok, phys = _consumer_page(
                    P, self.pt, req, True, c, self.width, self.pages
                )
                if not page_ok:
                    continue
                if v == "raw":
                    src[r, c] = self.raw_row(phys, c % P)
                    continue
                t0 = c // span * span
                if v == "r16":
                    wave, rblk = (r % 16) // 4, r // 16
                    t0w = rblk * 16 + wave * 4
                    first = next(
                        i
                        for i in range(4)
                        if t0w + i < self.rows and self.t2r[t0w + i] == req
                    )
                    key = (v, rblk, wave, first, t0)
                    if key not in cache:
                        cache[key] = self._stage(
                            "r16", req, t0, wave, rblk=rblk, p=first
                        )
                    lds, base = cache[key]
                    src[r, c] = self._tdm_row(lds, base, c - t0, ROW, v)
                elif v in ("r32", "fp8r32"):
                    rblk = r // 32
                    key = (v, rblk, t0)
                    if key not in cache:
                        lds = {}
                        for w in range(4):
                            lds.update(self._stage(v, req, t0, w, rblk=rblk)[0])
                        cache[key] = (lds, 0)
                    lds, base = cache[key]
                    pitch = ROW if v == "r32" else ROW8
                    if v == "r32":
                        src[r, c] = self._tdm_row(lds, base, c - t0, pitch, v)
                    else:
                        src[r, c] = lds[(c - t0) * pitch]
                elif v == "persist":
                    k = (c - t0) // 64
                    c0 = t0 + 64 * k
                    key = (v, req, c0)
                    if key not in cache:
                        lds = {}
                        for w in range(4):
                            lds.update(self._stage("persist", req, c0, w, buf=k % 2)[0])
                        cache[key] = (lds, (k % 2) * 64 * ROW)
                    lds, base = cache[key]
                    src[r, c] = self._tdm_row(lds, base, c - c0, ROW, "r32")
                else:  # draft / decg: 4 wave slabs of 32 columns
                    w = (c - t0) // 32
                    c0 = t0 + 32 * w
                    key = (v, req, c0)
                    if key not in cache:
                        cache[key] = self._stage(v, req, c0, w)
                    lds, base = cache[key]
                    src[r, c] = self._tdm_row(lds, base, c - c0, ROW, v)
        return src


def _scores(fx_, src):
    q = fx_["q"].float().cpu().numpy().astype(np.float64)  # [rows, 4, 128]
    k = fx_["compressed_k_cache"].float().cpu().numpy().astype(np.float64)
    if fx_.get("compressed_k_scale") is not None:
        k = k * fx_["compressed_k_scale"].float().cpu().numpy()[..., None]
    k = k.reshape(-1, 128)
    out = np.full(src.shape, -np.inf)
    rr, cc = np.nonzero(src >= 0)
    if rr.size:
        kk = k[src[rr, cc]]  # [n, 128]
        dots = np.einsum("nhd,nd->nh", q[rr], kk)
        out[rr, cc] = np.maximum(dots, 0).sum(-1) / math.sqrt(128)
    return out


def _fixture(
    P, regime, rows, requests, max_ctx, width, pages, seed=0, dtype="bf16", edits=None
):
    case = S.make_case(
        f"p{P}_{regime}",
        regime,
        rows=rows,
        requests=requests,
        max_ctx=max_ctx,
        index_dtype=dtype,
        table_width=width,
        cache_pages=pages,
        page_size=P,
    )
    d = G.build_case(case, device="cpu", seed=seed)
    if edits:
        edits(d)
    return d


def _golden(d, columns):
    # The frozen torch golden indexes ``keys[phys]`` after clamping only the lower bound, so
    # page ids >= pages raise on CPU. Contract-wise every out-of-range id is the same invalid
    # page as -1 (-inf, outside the compared mask): feed the golden the -1 spelling, keep the
    # out-of-range ids for the kernel emulation.
    pt = d["page_table"].clone()
    pt[pt >= d["compressed_k_cache"].shape[0]] = -1
    args = (
        d["q"],
        d["compressed_k_cache"],
        pt,
        d["token_to_request"],
        d["query_positions"],
        d["context_lens"],
    )
    return G.reference_logits(
        *args,
        compress_ratio=4,
        max_columns=columns,
        compressed_k_scale=d["compressed_k_scale"],
        dtype=torch.float64,
    )


def _edits_bad_pages(d):
    pt = d["page_table"]
    pages = d["compressed_k_cache"].shape[0]
    pt[0, 1] = -1  # hole
    pt[0, 2] = pages  # out of range
    pt[-1, 0] = pages + 1000  # out of range at the first page
    pt[-1, 3] = -7


def _edits_invalid_t2r(d):
    d["token_to_request"][0] = -1
    d["token_to_request"][-1] = 10**6


class AddressEmulation(unittest.TestCase):
    """Real producer/consumer helpers vs the frozen golden (see module docstring)."""

    CASES = [
        # (name, regime, rows, requests, max_ctx, width, pages, variants, edits)
        ("decode", "decode", 8, 2, 1000, 32, 80, ("decg", "r16", "raw"), None),
        (
            "decode_bad_pages",
            "decode",
            8,
            2,
            1500,
            32,
            80,
            ("decg", "r16", "raw"),
            _edits_bad_pages,
        ),
        ("draft", "draft", 3, 3, 2047, 40, 200, ("draft", "raw"), None),
        (
            "draft_bad_pages",
            "draft",
            3,
            3,
            2047,
            40,
            200,
            ("draft", "raw"),
            _edits_bad_pages,
        ),
        (
            "draft_invalid_t2r",
            "draft",
            4,
            4,
            900,
            16,
            64,
            ("draft",),
            _edits_invalid_t2r,
        ),
        (
            "prefill",
            "prefill",
            64,
            2,
            2300,
            40,
            120,
            ("r32", "fp8r32", "persist", "raw"),
            None,
        ),
        (
            "prefill_bad",
            "prefill",
            64,
            2,
            2300,
            40,
            120,
            ("r32", "persist"),
            _edits_bad_pages,
        ),
        ("narrow_table", "decode", 4, 1, 160, 3, 8, ("decg", "r16"), None),
    ]

    def _check(
        self,
        P,
        name,
        regime,
        rows,
        reqs,
        max_ctx,
        width,
        pages,
        variants,
        edits,
        columns=None,
    ):
        d = _fixture(P, regime, rows, reqs, max_ctx, width, pages, edits=edits)
        cols = width * P if columns is None else min(columns, width * P)
        gold, gvis, valid = _golden(d, cols)
        for v in variants:
            emu = Emu(d, P, v, cols)
            src = emu.run()
            out = torch.from_numpy(_scores(d, src)).to(torch.float32)
            vis = torch.from_numpy(emu.vis.astype(np.int32))
            res = G.compare_logits(
                out, vis, gold.to(torch.float32), gvis, valid, rtol=1e-5, atol=1e-5
            )
            self.assertTrue(res["pass"], (P, name, v, res))
            inside = valid & (
                torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None]
            )
            self.assertTrue(bool(torch.isneginf(out[~inside]).all()), (P, name, v))
            # every in-mask element is exactly the golden's K row (address identity)
            ins = inside.numpy()
            rr, cc = np.nonzero(ins)
            pt = d["page_table"].numpy().astype(np.int64)
            want = pt[emu.sreq[rr], cc // P] * P + cc % P
            self.assertTrue(np.array_equal(src[rr, cc], want), (P, name, v))
            self.assertTrue((src[~ins] == -1).all(), (P, name, v))

    def test_page16_all_regimes(self):
        for case in self.CASES:
            with self.subTest(case=case[0]):
                self._check(16, *case)

    def test_page4_harness_sanity(self):
        """The same harness on the GPU-proven P = 4 code paths (self-check of the emulator)."""
        for name, regime, rows, reqs, ctx, width, pages, variants, edits in self.CASES:
            with self.subTest(case=name):
                self._check(
                    4,
                    name,
                    regime,
                    rows,
                    reqs,
                    ctx,
                    width * 4,
                    pages * 4,
                    variants,
                    edits,
                )

    def test_page16_visible_below_columns_and_tails(self):
        # columns = pow2 bucket below W*16, not a multiple of the 128/512-column spans
        self._check(
            16,
            "tail_cols",
            "prefill",
            40,
            1,
            700,
            64,
            64,
            ("r32", "r16", "raw"),
            None,
            columns=1000,
        )
        self._check(
            16, "short_ctx", "decode", 4, 1, 37, 8, 8, ("decg", "r16", "raw"), None
        )

    def test_page16_gather_lookups(self):
        """P = 16: every 8-row gather costs exactly one page-table s_buffer_load; P = 4: two."""
        for P, per in ((16, 1), (4, 2)):
            d = _fixture(P, "draft", 1, 1, 256, 256 // P, 64)
            for kind in ("draft", "decg", "r32", "persist", "r16"):
                emu = Emu(d, P, kind, 64)
                emu._stage(kind, 0, 0, 0)
                self.assertGreater(emu.stats["gathers"], 0)
                self.assertEqual(
                    emu.stats["lookups"],
                    per * emu.stats["gathers"],
                    (P, kind, emu.stats),
                )

    def test_p16_row_ids_clamps(self):
        env = Env()
        f = _exec(env, 16, "", "_p16_row_ids")
        pt = Plane([[5, -1, 99, 3]])
        rows = [_i(x) for x in f(pt, FI(0), FI(4), FI(10), FI(0), 8)]
        self.assertEqual(rows, list(range(5 * 16 + 8, 5 * 16 + 16)))
        for logical in (1, 2, 4, 7):  # hole, page >= pages, logical >= width (x2)
            rows = [_i(x) for x in f(pt, FI(0), FI(4), FI(10), FI(logical), 0)]
            self.assertEqual(rows, list(range(8)), logical)
        rows = [_i(x) for x in f(pt, FI(0), FI(4), FI(10), FI(3), 8, FB(False))]
        self.assertEqual(rows, list(range(8, 16)))  # invalid request -> page 0


class SourceShape(unittest.TestCase):
    """P = 4 code is unchanged; P = 16 is a compile-time branch everywhere."""

    def _base(self):
        try:
            return subprocess.check_output(
                ["git", "show", f"{BASE_REV}:{KREL}"], cwd=ROOT
            ).decode()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"git base {BASE_REV} unavailable: {exc}")

    def test_base_sha(self):
        import hashlib

        self.assertEqual(hashlib.sha256(self._base().encode()).hexdigest(), BASE_SHA)

    def test_page4_tdm_sites_unchanged(self):
        """Each `if P == 4:` branch == the base site with PAGE_SIZE spelled as 4."""
        base = self._base()
        n = 0
        sites = [
            n
            for b in (
                "_build_wmma_kernel",
                "_build_wmma_fp8_kernel",
                "_build_draft_kernel",
                "_build_decode_group_kernel",
            )
            for n in ast.walk(_builder(b))
        ]
        for node in sites:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "P == 4":
                code = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
                legacy = (
                    re.sub(r"// 4\b", "// PAGE_SIZE", code)
                    .replace("range_constexpr(4)", "range_constexpr(PAGE_SIZE)")
                    .replace("phys * 4 + ", "phys * PAGE_SIZE + ")
                )
                base_norm = ast.unparse(ast.parse(base))
                self.assertIn("PAGE_SIZE", legacy)
                for line in legacy.splitlines():
                    self.assertIn(line.strip(), base_norm, line)
                n += 1
        # r16, pair bf16, persist, pair fp8, draft, decg (+ ROUND-R2C ktile producers bf16
        # 8-wave / fp8, whose P = 4 bodies are the persist / pair-fp8 sites line by line;
        # + ROUND-R2D lean ktile prologues bf16 / fp8 ``_ktl_pre``)
        self.assertEqual(n, 10)

    def test_page4_wrapper_passes_no_new_keyword(self):
        self.assertIn("pg_kw = _NO_KW if page == PAGE_SIZE else _PAGE_KW[page]", SRC)
        self.assertEqual(_const("COMPRESS_RATIO"), 4)
        self.assertEqual(
            tuple(_const("SUPPORTED_PAGE_SIZES")), (4, 8, 16)
        )  # ROUND-R2: + P = 8

    def test_visibility_never_divides_by_page(self):
        for line in SRC.splitlines():
            if (
                "vis" in line
                and "//" in line
                and "pos + 1" in line
                and "`" not in line
                and not line.strip().startswith("#")
            ):
                self.assertIn("COMPRESS_RATIO", line, line)
        self.assertNotIn("// PAGE_SIZE", SRC.split("def qsa_pad_plan")[0])

    def test_builders_take_page_size(self):
        for b in (
            "_build_kernel",
            "_build_wmma_kernel",
            "_build_wmma_fp8_kernel",
            "_build_draft_kernel",
            "_build_decode_group_kernel",
        ):
            node = _builder(b)
            args = [a.arg for a in node.args.args]
            self.assertIn("page_size", args, b)  # ROUND-R2 knobs follow page_size
            self.assertIn(
                "P = _check_page_size(page_size)", ast.get_source_segment(SRC, node), b
            )


# ---------------------------------------------------------------------------
# Op / profile
# ---------------------------------------------------------------------------
_N = [0]


def load_pkg():
    _N[0] += 1
    name = f"qsa_p16_test{_N[0]}"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules[name + ".api"], sys.modules[name + ".profiles"]


def load_gen():
    path = os.path.join(PKG_DIR, "gen_profile.py")
    _, _, P = load_pkg()
    spec = importlib.util.spec_from_file_location("qsa_p16_gen", path)
    g = importlib.util.module_from_spec(spec)
    sys.modules["profiles"] = P
    spec.loader.exec_module(g)
    return g, P


class OpProfile(unittest.TestCase):
    def setUp(self):
        self.g, self.P = load_gen()

    def _rules(self):
        caps = self.g.capture_shapes_p16()
        return [self.g.make_rule(s, caps) for s in self.g.RULE_SPEC_P16]

    def test_p16_rules_mirror_p4(self):
        p4 = {s["id"]: s for s in self.g.RULE_SPEC}
        for s in self.g.RULE_SPEC_P16:
            base = p4[s["id"][: -len("_p16")]]
            self.assertEqual(s["page_size"], 16)
            for k in ("relation", "rows", "requests", "columns", "cache_dtype"):
                self.assertEqual(s[k], base[k], (s["id"], k))
            self.assertLessEqual(s["table_width"]["max"], 4096)
            self.assertLessEqual(
                self.P.column_values(s["columns"])[0], s["table_width"]["max"] * 16
            )

    def test_p16_rules_validate_and_are_pending(self):
        for r in self._rules():
            self.P.validate_rule(r)
            self.assertEqual(r["golden"]["status"], "pending")
            self.assertTrue(r["vid"].startswith("qsa.page16."), r["vid"])
            self.assertGreater(len(r["samples"]), 3, r["id"])
            for s in r["samples"]:
                self.assertLessEqual(s["columns"], s["table_width"] * 16)

    def test_e2e_block64_shapes_covered(self):
        rules = [self.P.ProfileRule(r) for r in self._rules()]
        for k in self.g.agentx_p16_shapes():
            hits = [r.id for r in rules if r.admits(*k)]
            self.assertEqual(len(hits), 1, (k, hits))

    def test_page_isolation_and_reasons(self):
        prof4 = self.P.load_profile(
            os.path.join(
                ROOT,
                "aiter",
                "configs",
                "qsa_index_scorer",
                "qsa_page4_gfx1250_v1.json",
            ),
            use_cache=False,
        )
        data = json.loads(json.dumps(prof4.data))
        data["rules"] = [r for r in data["rules"] if self.P.page_of(r) == 4]
        # treat every p16 rule as verified for the matcher test
        rules = self._rules()
        for r in rules:
            r["golden"] = {
                "status": "pass",
                "kernel_sha256": data["kernel"]["sha256"],
                "checked": [dict(s, **{"pass": True}) for s in r["samples"]],
            }
        data["rules"] = data["rules"] + rules
        prof = self.P.profile_from_data(data)
        kw = {"arch": "gfx1250", "cu": 256, "cache_dtype": "bf16"}
        m = self.P.match(
            prof,
            rows=64,
            requests=16,
            columns=65536,
            table_width=4096,
            page_size=16,
            **kw,
        )
        self.assertTrue(m.ok and m.row.id == "decode_bf16_p16", m)
        m4 = self.P.match(
            prof, rows=64, requests=16, columns=65536, table_width=16384, **kw
        )
        self.assertTrue(m4.ok and "p16" not in m4.row.cell, m4)
        # a page-16 call never matches a page-4 rule (W=16384 is outside every p16 rule)
        m = self.P.match(
            prof,
            rows=64,
            requests=16,
            columns=65536,
            table_width=16384,
            page_size=16,
            **kw,
        )
        self.assertFalse(m.ok)
        self.assertEqual(m.reason.value, "table_width_uncovered")
        m = self.P.match(
            prof,
            rows=64,
            requests=16,
            columns=65536,
            table_width=4096,
            page_size=32,
            **kw,
        )
        self.assertEqual(m.reason.value, "cache_shape_unsupported")
        # R2: page 8 is a profile page size; with no page-8 rule in this profile it is uncovered
        m = self.P.match(
            prof,
            rows=64,
            requests=16,
            columns=65536,
            table_width=4096,
            page_size=8,
            **kw,
        )
        self.assertEqual(m.reason.value, "rows_rule_uncovered")
        # pending p16 rules are never selected
        data2 = json.loads(json.dumps(prof4.data))
        data2["rules"] = [r for r in data2["rules"] if self.P.page_of(r) == 4]
        data2["rules"] += self._rules()
        m = self.P.match(
            self.P.profile_from_data(data2),
            rows=64,
            requests=16,
            columns=65536,
            table_width=4096,
            page_size=16,
            **kw,
        )
        self.assertEqual(m.reason.value, "row_unverified")

    def test_page4_signatures_unchanged(self):
        prof4 = json.load(
            open(  # noqa: SIM115 -- read once, closed by GC
                os.path.join(
                    ROOT,
                    "aiter",
                    "configs",
                    "qsa_index_scorer",
                    "qsa_page4_gfx1250_v1.json",
                ),
                encoding="utf-8",
            )
        )
        for row in prof4["rows"]:
            self.assertEqual(self.P.row_signature(dict(row, page_size=4)), row["sig"])
        for rule in prof4["rules"]:
            self.assertEqual(self.P.rule_signature(rule), rule["sig"])

    def test_variant_names_and_prewarm_geometry(self):
        _, api, P = load_pkg()
        for regime, cols in (
            ("draft", 65536),
            ("decode", 65536),
            ("tail", 4096),
            ("prefill", 256),
            ("prefill", 65536),
        ):
            v4, p4 = P.expected_variant("bf16", regime, cols)
            v16, p16 = P.expected_variant("bf16", regime, cols, 16)
            self.assertEqual(v16, v4.replace("qsa.page4.", "qsa.page16."))
            self.assertEqual(p16, p4)
            spec4 = api._variant_spec(v4, p4, False)
            self.assertEqual(api._variant_spec(v16, p16, False), spec4)

    def test_layout_reason_for_other_pages(self):
        _, api, P = load_pkg()
        prof = P.load_profile(
            os.path.join(
                ROOT,
                "aiter",
                "configs",
                "qsa_index_scorer",
                "qsa_page4_gfx1250_v1.json",
            ),
            use_cache=False,
        )
        for page, want in (
            (2, "cache_shape_unsupported"),
            (32, "cache_shape_unsupported"),
        ):
            cache = torch.zeros((4, page, 1, 128), dtype=torch.bfloat16)
            d = api._layout_decision(cache, None, cache.device, prof)
            self.assertEqual(d.reason.value, want)
        for page in (4, 8, 16):  # R2-MERGE: the kernel implements page 8
            raw = torch.zeros(4 * page * 128 * 2 + 128, dtype=torch.uint8)
            off = (-raw.data_ptr()) % 128
            cache = (
                raw[off : off + 4 * page * 128 * 2]
                .view(torch.bfloat16)
                .view(4, page, 1, 128)
            )
            self.assertEqual(
                api._layout_decision(cache, None, cache.device, prof), "bf16"
            )


if __name__ == "__main__":
    unittest.main()

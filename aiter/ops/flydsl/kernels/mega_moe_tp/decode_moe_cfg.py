# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuning configs and kernel names of the decode-shape (8-128 row) MoE chain: E128 top-4, H6144 / I768, MXFP4
routed experts plus an MXFP8 shared expert (MiniMax-M3.1 at TP4).

Every knob the chain's kernels take is a field here, not a per-row-count table: the fmoe tuner enumerates
``expert_candidates`` / ``router_candidates``, times them through ``fused_moe``, and writes the winner's *name* into
a tuned CSV row; ``fused_moe`` and the router entry parse the name back. The default configs reproduce the former
hard-coded per-row-count tiers, so a shape without a row runs exactly what it ran before.

* ``ExpertCfg`` (the whole-graph implementation ``impl__flydsl_m3decode__<name>`` of ``kernelName1``): the routed /
  shared tile knobs of the GEMM1 + GEMM2 pair (``decode_expert_gemms``).
* ``RouterCfg`` (``kernelName2`` of the same row, ``flydsl_m3route_<name>``): the split-K router GEMM
  (``decode_router_gemm*``) and where the input quant runs.

Pure Python, no torch / flydsl import, so names and candidates can be built on a machine without a GPU stack.
"""

import re
from dataclasses import dataclass, replace

H, I, E, TOPK = 6144, 768, 128, 4
# gfx950 LDS budget of a workgroup (same value as ``mega_moe_tp.LDS_LIMIT``; not imported to keep this module CPU-only)
LDS_LIMIT = 160 * 1024
ROWS = (8, 128)  # rows the expert tiles cover (the router covers 1-128)
ROUTER_MAX_ROWS = 128
IMPL_NAME = "flydsl_m3decode"
ROUTE_PREFIX = "flydsl_m3route_"

# ---------------------------------------------------------------- expert tiles


@dataclass(frozen=True)
class ExpertCfg:
    """Tile knobs of the routed + shared-expert GEMM pair. All outputs are bit-identical for every legal value
    (the K order is kept), so these only trade latency.

    sh_bm: rows per shared gate/up tile; sh_kw: its K-waves; sh_pf: its prefetch depth; sh_tn: columns per shared
    down tile; g1_pf: routed GEMM1 prefetch; g2_tn: routed GEMM2 columns per tile; g2_pf: its prefetch;
    g1_prio: GEMM1 prologue issue priority ``(a, b)`` or None.
    """

    sh_bm: int
    sh_kw: int
    sh_pf: int
    sh_tn: int
    g1_pf: int
    g2_tn: int
    g2_pf: int
    g1_prio: tuple | None = None

    _RE = re.compile(
        r"sbm(?P<sh_bm>\d+)_skw(?P<sh_kw>\d+)_spf(?P<sh_pf>\d+)_stn(?P<sh_tn>\d+)"
        r"_g1pf(?P<g1_pf>\d+)_g2tn(?P<g2_tn>\d+)_g2pf(?P<g2_pf>\d+)_prio(?P<prio>n|\d\d)"
    )

    def name(self) -> str:
        prio = "n" if self.g1_prio is None else f"{self.g1_prio[0]}{self.g1_prio[1]}"
        return (
            f"sbm{self.sh_bm}_skw{self.sh_kw}_spf{self.sh_pf}_stn{self.sh_tn}"
            f"_g1pf{self.g1_pf}_g2tn{self.g2_tn}_g2pf{self.g2_pf}_prio{prio}"
        )

    @classmethod
    def parse(cls, name: str) -> "ExpertCfg":
        m = cls._RE.fullmatch(name)
        if m is None:
            raise ValueError(f"not a decode expert config name: {name!r}")
        g = m.groupdict()
        prio = None if g.pop("prio") == "n" else (int(m["prio"][0]), int(m["prio"][1]))
        cfg = cls(**{k: int(v) for k, v in g.items()}, g1_prio=prio)
        if not cfg.legal():
            raise ValueError(f"illegal decode expert config: {name!r}")
        return cfg

    def to_tiles(self) -> dict:
        """The ``tiles`` dict of ``decode_expert_gemms.get_plan``."""
        tiles = {
            "sh_bm": self.sh_bm,
            "sh_kw": self.sh_kw,
            "sh_pf": self.sh_pf,
            "sh_tn": self.sh_tn,
            "g1_pf": self.g1_pf,
            "g2_tn": self.g2_tn,
            "g2_pf": self.g2_pf,
        }
        if self.g1_prio is not None:
            tiles["g1_prio"] = self.g1_prio
        return tiles

    def legal(self) -> bool:
        """Static legality (n-independent): every value the tile bodies were written and verified for."""
        return (
            self.sh_bm in (16, 32, 64)
            and self.sh_kw in (2, 4)
            and 1 <= self.sh_pf <= 4
            # measured: a 64-row shared tile with prefetch >= 3 is not bit-identical (n = 128)
            and not (self.sh_bm == 64 and self.sh_pf >= 3)
            and self.sh_tn in (64, 128, 256)
            and 1 <= self.g1_pf <= 6
            and self.g2_tn in (128, 256)
            and 1 <= self.g2_pf <= 3
            and (self.g1_prio is None or tuple(self.g1_prio) == (3, 0))
        )

    @staticmethod
    def supports_rows(n: int) -> bool:
        return ROWS[0] <= n <= ROWS[1]

    @classmethod
    def default(cls, n: int) -> "ExpertCfg":
        """The former per-row-count tiling (``tiles_for`` + the shared-tile helpers), every field explicit."""
        # n <= 96: 16-row shared tiles with 4 K-waves; <= 128: 2 K-waves (``shared_fold_tile_for``)
        sh_bm, sh_kw, sh_pf = (16, 4, 2) if n <= 96 else (16, 2, 2)
        sh_tn = 64 if n <= 16 else 256  # ``shared_g2_columns_for``
        g1_pf, g2_tn, g2_pf = 3, 256, 2
        if n <= 16:
            if n == 8:
                sh_pf = 3
            else:
                g1_pf, g2_tn = 6, 128
        elif n <= 64:
            sh_tn, sh_bm = 128, 32
        else:
            sh_tn, sh_bm, sh_pf = 128, 64, 1
        prio = None
        if 32 <= n <= 128:  # ``ASM_ROWS``
            g2_pf, prio = 3, (3, 0)
        return cls(sh_bm, sh_kw, sh_pf, sh_tn, g1_pf, g2_tn, g2_pf, prio)


# one-at-a-time neighbours of a config: the staged search around the default
_EXPERT_AXES = {
    "sh_bm": (16, 32, 64),
    "sh_kw": (2, 4),
    "sh_pf": (1, 2, 3),
    "sh_tn": (64, 128, 256),
    "g1_pf": (2, 3, 4, 6),
    "g2_tn": (128, 256),
    "g2_pf": (2, 3),
    "g1_prio": (None, (3, 0)),
}


def expert_candidates(n: int, full: bool = False) -> list:
    """Configs for ``n`` rows, the default first. Staged by default (the default plus every one-knob change, ~15
    configs); ``full`` is the cross product of the axes (several hundred; for an overnight sweep).
    """
    base = ExpertCfg.default(n)
    out = [base]
    if full:
        import itertools

        for vals in itertools.product(*_EXPERT_AXES.values()):
            cfg = ExpertCfg(*vals)
            if cfg.legal() and cfg not in out:
                out.append(cfg)
        return out
    for axis, values in _EXPERT_AXES.items():
        for v in values:
            cfg = replace(base, **{axis: v})
            if cfg.legal() and cfg not in out:
                out.append(cfg)
    return out


# ---------------------------------------------------------------- router GEMM


@dataclass(frozen=True)
class RouterCfg:
    """The split-K router GEMM (x bf16 [m, 6144] x gate fp32 [128, 6144], bf16x3) and the input-quant placement.

    kind: ``direct`` (1-16 rows, no LDS, line-layout gate copy), ``lines`` (17-32 rows, line-layout gate copy) or
    ``nblocks`` (any rows, the generic tile); split_k / block_k: the K split and the K per step (the partials'
    sum order); waves: waves per workgroup (each takes whole 16-row m-tiles); nbw: n-blocks per workgroup; quant: 0 the quant stays in the
    top-k launch, 1 appended / 2 prepended to the router launch; x_direct: the four-wave kernel reads x directly.
    ``direct`` / ``lines`` ignore waves / nbw / quant / x_direct (kept 0 / False for a canonical name).
    """

    kind: str
    split_k: int
    block_k: int
    waves: int = 0
    nbw: int = 0
    quant: int = 0
    x_direct: bool = False

    _RE = re.compile(
        r"(?P<kind>direct|lines|nblocks)_s(?P<split_k>\d+)_bk(?P<block_k>\d+)"
        r"(?:_w(?P<waves>\d+)_nb(?P<nbw>\d+)_q(?P<quant>\d)(?P<xd>_xd)?)?"
    )

    def name(self) -> str:
        s = f"{self.kind}_s{self.split_k}_bk{self.block_k}"
        if self.kind == "nblocks":
            s += f"_w{self.waves}_nb{self.nbw}_q{self.quant}" + (
                "_xd" if self.x_direct else ""
            )
        return ROUTE_PREFIX + s

    @classmethod
    def parse(cls, name: str) -> "RouterCfg":
        if not name.startswith(ROUTE_PREFIX):
            raise ValueError(f"not a decode router config name: {name!r}")
        m = cls._RE.fullmatch(name[len(ROUTE_PREFIX) :])
        if m is None:
            raise ValueError(f"not a decode router config name: {name!r}")
        g = m.groupdict()
        nb = g["kind"] == "nblocks"
        if nb != (g["waves"] is not None):
            raise ValueError(f"not a decode router config name: {name!r}")
        return cls(
            kind=g["kind"],
            split_k=int(g["split_k"]),
            block_k=int(g["block_k"]),
            waves=int(g["waves"] or 0),
            nbw=int(g["nbw"] or 0),
            quant=int(g["quant"] or 0),
            x_direct=g["xd"] is not None,
        )

    @property
    def uses_lines(self) -> bool:
        """Reads the line-layout copy of the gate (``router_line_layout``)."""
        return self.kind in ("direct", "lines")

    def token_major(self, m: int) -> bool:
        """Partials are written token-major [m, split_k, 128] (the top-k reads constant split offsets): up to 64
        rows, and always when the quant rides in the router launch."""
        return m <= 64 or self.quant != 0

    def fits(self, m: int) -> bool:
        """Static legality for ``m`` rows: the compile-time asserts of ``decode_router_gemm_nblocks`` /
        ``_direct`` / ``_lines`` and the row coverage."""
        if not (1 <= m <= ROUTER_MAX_ROWS):
            return False
        # positive bounds first: names parse to zero split / block values, which must not reach a modulo
        if not (0 < self.split_k <= 96 and self.block_k > 0):
            return False
        if self.split_k % 8 or H % self.split_k:
            return False
        kc = H // self.split_k
        if self.block_k % 32 or kc % self.block_k or not 32 <= self.block_k <= 128:
            return False
        if self.kind == "direct":
            # 16-row tiles, no LDS; the kernel's fixed split-K (compile-time assert) and its verified block_k
            return (
                m <= 16
                and self.split_k == 48
                and self.waves == 0
                and self.block_k <= 64
            )
        if self.kind == "lines":
            return 17 <= m <= 32 and self.split_k == 48 and self.block_k == 64
        if self.nbw not in (1, 2, 4, 8) or self.quant not in (0, 1, 2):
            return False
        if self.waves not in (2, 4, 8):
            return False
        # W_PIECES / X_PIECES of the tile: whole 16-byte pieces per lane
        w_dwords = self.nbw * 16 * self.block_k // 4
        if w_dwords % (64 * self.waves) or 16 * self.block_k // 8 // 64 < 1:
            return False
        # measured: nbw >= 4 at 65-128 rows faults the GPU (the weight tile outgrows the LDS next to the x tile)
        if m > 64 and self.nbw > 2:
            return False
        return self.lds_bytes() <= LDS_LIMIT and not (
            self.x_direct and not 33 <= m <= 57
        )

    def lds_bytes(self) -> int:
        """LDS of the ``nblocks`` kernel (``decode_router_gemm_nblocks.Shared``: weight ring + double-buffered x)."""
        spb = self.block_k // 32
        nstep = H // self.split_k // self.block_k
        return (self.nbw * nstep * 3 * spb * 64 * 4 + 2 * self.waves * spb * 64 * 4) * 4

    @classmethod
    def default(cls, m: int) -> "RouterCfg":
        """The former tier table (``decode_router_gemm`` / ``decode_router_chain``)."""
        split_k, block_k = (48, 64) if m <= 64 else (16, 128)
        quant = 1 if 58 <= m <= 64 else 2 if m > 64 else 0
        if quant == 0:
            if m <= 16:
                return cls("direct", 48, 64)
            if m <= 32:
                return cls("lines", 48, 64)
        waves = 2 if m <= 32 else 8 if (m + 15) // 16 > 8 else 4
        return cls("nblocks", split_k, block_k, waves, 1, quant, x_direct=33 <= m <= 57)


def router_partials_default(m: int) -> RouterCfg:
    """``RouterCfg.default`` with the quant left to the top-k launch (the partials-only router launch)."""
    return replace(RouterCfg.default(m), quant=0)


_ROUTER_SPLITS = ((8, 64), (16, 64), (16, 128), (24, 64), (32, 64), (48, 64), (48, 128))


def router_candidates(m: int, full: bool = False) -> list:
    """Router configs for ``m`` rows (default first). Staged: the default plus one-knob changes (split-K / block-K
    pairs, waves, n-blocks per workgroup, quant placement, x_direct, and the direct / lines kernels where they
    apply); ``full``: the cross product."""
    base = RouterCfg.default(m)
    out = [base]

    def add(cfg):
        if cfg.fits(m) and cfg not in out:
            out.append(cfg)

    nb = base if base.kind == "nblocks" else RouterCfg("nblocks", 48, 64, 2, 1, 0)
    if full:
        # every legal (split_k, block_k): the split-K factor is a tuning field, so none is skipped
        for s in range(8, 97, 8):
            for bk in (32, 64, 96, 128):
                for w in (2, 4, 8):
                    for nbw in (1, 2, 4, 8):
                        for q in (0, 1, 2):
                            for xd in (False, True):
                                add(RouterCfg("nblocks", s, bk, w, nbw, q, xd))
    else:
        for s, bk in _ROUTER_SPLITS:
            add(replace(nb, split_k=s, block_k=bk))
        for w in (2, 4, 8):
            add(replace(nb, waves=w))
        for nbw in (1, 2, 4, 8):
            add(replace(nb, nbw=nbw))
        for q in (0, 1, 2):
            add(replace(nb, quant=q))
        add(replace(nb, x_direct=not nb.x_direct))
    add(RouterCfg("direct", 48, 64))
    add(RouterCfg("lines", 48, 64))
    return out


# ---------------------------------------------------------------- whole-row config


def impl_kernel_name(cfg: ExpertCfg) -> str:
    """``kernelName1`` of a tuned row: the whole-graph fused_moe implementation and its config."""
    return f"impl__{IMPL_NAME}__{cfg.name()}"


def parse_impl_kernel_name(kernel_name: str) -> ExpertCfg:
    prefix = f"impl__{IMPL_NAME}__"
    if not kernel_name.startswith(prefix):
        raise ValueError(f"not a decode-chain implementation name: {kernel_name!r}")
    return ExpertCfg.parse(kernel_name[len(prefix) :])

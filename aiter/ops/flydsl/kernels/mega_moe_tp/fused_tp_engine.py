# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel TP MegaMoE (:mod:`.fused_tp`).

One launch per forward: quantize + AllGather, route scan, GEMM1, activation,
MXFP4 requant, GEMM2, top-k reduce and ReduceScatter all run inside one FlyDSL
kernel, and the GEMM1 -> GEMM2 intermediate stays in LDS.

Per-call launch config (:class:`LaunchCfg`: row tile ``block_m``, weight ring
depth, dynamic vs static schedule, route-row format, column split) comes from
``aiter/configs/model_configs/mega_moe_tp_a4w4_tuned.csv`` (tuned per model,
TP, comm mode and token count by ``op_tests/multigpu_tests/tune_mega_moe_tp.py``;
``AITER_MEGAMOE_TP_CONFIG`` points elsewhere), else from a heuristic.
"""

from __future__ import annotations

import functools
import heapq
import math
import os
from dataclasses import dataclass

import torch

from ..tensor_shim import _run_compiled
from .fused_tp import (
    CTRL_ERR,
    CTRL_INTS,
    FLAG_INTS,
    MAX_TP,
    NCTA_MAX,
    TL_SLOTS,
    UNIT_G1X,
    UNIT_REC,
    UNIT_G2COL,
    UNIT_SIGNAL,
    XQ_MAX,
    XQ_SHIFT,
    XQ_P,
    compile_fused_tp,
    fused_tp_consts,
    gemm2_chunk_groups,
    gemm2_group_step,
    fused_tp_supported,
)
from .symmetric_arena import SymmetricArena

__all__ = ["FusedTpMegaMoe", "LaunchCfg", "fused_tp_supported", "tuned_config_path"]

LDS_LIMIT = 160 * 1024
TUNED_CSV = "mega_moe_tp_a4w4_tuned.csv"
# the key columns of a tuned row, and its config columns
CSV_KEY = ("gfx", "cu_num", "tp", "comm_mode", "model_dim", "inter_dim", "expert", "topk", "act")
CSV_CFG = ("block_m", "nsk", "dyn", "route_fp8", "xsplit", "ll", "llr")


@dataclass(frozen=True)
class LaunchCfg:
    """One launch's variant: row tile block_m = 16 * mt, weight ring depth
    nsk (k-steps), dynamic schedule, E4M3 (else bf16) route rows, column
    split of the static schedule's leftover experts, LL ReduceScatter
    packets (data + tag: no completion flags; twice the wire bytes), and
    (llr, with ll) LL route rows pushed as soon as they land -- no chunk
    counting -- with an LL routing all-gather (AG/RS layer) or an LL output
    all-reduce (AR/AR layer)."""

    mt: int
    nsk: int
    dyn: bool
    route_fp8: bool
    xsplit: bool
    ll: bool = False
    llr: bool = False

    @property
    def block_m(self) -> int:
        return 16 * self.mt


def tuned_config_path() -> str:
    return os.environ.get("AITER_MEGAMOE_TP_CONFIG") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "configs", "model_configs", TUNED_CSV,
    )


@functools.cache
def _tuned_rows(path: str) -> dict:
    """key -> sorted [(token, LaunchCfg)] of a tuned CSV (empty if absent)."""
    import csv

    rows: dict = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            key = tuple(str(r[k]) for k in CSV_KEY)
            cfg = LaunchCfg(
                mt=int(r["block_m"]) // 16,
                nsk=int(r["nsk"]),
                dyn=int(r["dyn"]) == 1,
                route_fp8=int(r["route_fp8"]) == 1,
                xsplit=int(r["xsplit"]) == 1,
                ll=int(r.get("ll") or 0) == 1,
                llr=int(r.get("llr") or 0) == 1,
            )
            rows.setdefault(key, []).append((int(r["token"]), cfg))
    for v in rows.values():
        v.sort(key=lambda t: t[0])
    return rows


class _Sched:
    """A static work schedule: the unit table, the per-CTA records, and the
    compile-time split parameters it needs."""

    def __init__(self, units, recs, npieces, piece_e0, xsplit, xrem, xw, col0=0, ncol=0):
        self.units, self.recs = units, recs
        self.npieces, self.piece_e0 = npieces, piece_e0
        self.xsplit, self.xrem, self.xw = xsplit, xrem, xw
        # column slices claimed at run time: units[col0 : col0 + ncol]
        self.col0, self.ncol = col0, ncol


class FusedTpMegaMoe:
    """AG + GEMM1 + act + GEMM2 + RS in one kernel, for one TP rank."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        max_local_tokens: int,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        activation: str = "silu",
        situ_beta: float = 1.0,
        situ_linear_beta: float = 1.0,
        comm_mode: str = "ag_rs",
        group=None,
        device: torch.device | None = None,
    ):
        if not fused_tp_supported(model_dim, inter_dim, world_size):
            raise ValueError(
                f"fused TP MegaMoE does not tile h{model_dim} i{inter_dim} tp{world_size}"
            )
        if comm_mode not in ("ag_rs", "ar_ar"):
            raise ValueError(f"unknown comm_mode {comm_mode!r}")
        self.ar = comm_mode == "ar_ar"
        self.rank, self.tp = int(rank), int(world_size)
        self.H, self.I, self.E, self.K = model_dim, inter_dim, experts, topk
        self.mmax = int(max_local_tokens)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.activation = activation
        self.situ = (float(situ_beta), float(situ_linear_beta))
        u8 = lambda t: t.view(torch.uint8)
        self.w1, self.w1s, self.w2, self.w2s = (
            u8(w1),
            u8(w1_scale),
            u8(w2),
            u8(w2_scale),
        )

        tot = self.mmax * self.tp
        H, K = model_dim, topk
        arena = SymmetricArena(group=group, device=self.device)
        self._x = arena.reserve("x", (tot, H // 2), torch.uint8)
        self._xs = arena.reserve("xs", (tot, H // 32), torch.uint8)
        # (16 B per route: LL routing packets, see compile_fused_tp)
        self._ids = arena.reserve("ids", (tot, 4 * K), torch.int32)
        self._w = arena.reserve("w", (tot, K), torch.float32)
        # ReduceScatter receive slots [source rank][row][H]. A peer only writes
        # launch n + 1's rows after this rank joined n + 1's AllGather.
        # (AR: every rank's partial of every row -- the LL output all-reduce)
        self._recv = arena.reserve(
            "recv", (self.tp, tot if self.ar else self.mmax, H), torch.bfloat16
        )
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        # AR/AR: the peers' partials of this rank's input shard, and the
        # all-gathered output (returned as a view: valid until the next call)
        self._pre = arena.reserve("pre", (self.tp, self.mmax, H) if self.ar else (1,), torch.bfloat16)
        self._yall = arena.reserve("yall", (tot, H) if self.ar else (1,), torch.bfloat16)
        arena.commit()
        self.arena = arena
        self.ctrl = torch.zeros(CTRL_INTS, dtype=torch.int32, device=self.device)
        # Per-route GEMM2 rows (weighted, bf16); the comm waves reduce them.
        self.routes = torch.empty(
            (tot * topk + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.y = torch.empty((self.mmax, H), dtype=torch.bfloat16, device=self.device)

        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        # every CTA sends ceil(mmax / n_cta) of this rank's tokens in the AllGather
        self.agr = max(1, -(-self.mmax // self.n_cta))
        if self.n_cta > NCTA_MAX:
            raise ValueError(f"at most {NCTA_MAX} CTAs (AllGather flag layout)")
        # the static schedule with and without column-split leftover experts
        self._scheds = {xs: self._schedule(xs) for xs in (True, False)}
        # K-slice partials of split experts' pieces 1..P-1, one route-row set each
        # (piece 0 writes the route's own row); the push adds them up.
        extra = max(max(sc.npieces - 1, 0) for sc in self._scheds.values()) * tot * topk
        # dynamic schedule (small batches, see compile_fused_tp dyn_e): up to
        # one partial region per inter slice
        self.dyn_max = min(tot, int(os.environ.get("AITER_MEGAMOE_DYN_MAX", "256")))
        if self.dyn_max > 0:
            extra = max(extra, (inter_dim // 128 - 1) * self.dyn_max * topk)
        if extra * H * 2 >= 1 << 31:
            raise ValueError("split-expert partial rows exceed 32-bit buffer offsets")
        self.proutes = torch.empty(
            (extra + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self._args: dict = {}
        # column-split leftover experts: their intermediate, by route index
        self.xg = (
            torch.empty((tot * topk * (inter_dim // 2 + inter_dim // 32),), dtype=torch.uint8, device=self.device)
            if any(sc.xsplit for sc in self._scheds.values())
            else None
        )
        # AITER_MEGAMOE_TIMELINE=1: per-CTA phase timestamps (s_memrealtime)
        self.tl = (
            torch.zeros((self.n_cta, TL_SLOTS), dtype=torch.int64, device=self.device)
            if os.environ.get("AITER_MEGAMOE_TIMELINE", "0") == "1"
            else None
        )
        # E4M3 ReduceScatter wire format: half the xGMI bytes, error vs torch
        # ~0.038 (split path: 0.034-0.044); AITER_MEGAMOE_RS_FP8=0: bf16.
        self.rs_fp8 = os.environ.get("AITER_MEGAMOE_RS_FP8", "1") == "1"
        key = (self.gfx, self.n_cta, self.tp, "ar_ar" if self.ar else "ag_rs",
               self.H, self.I, self.E, self.K, activation)
        self._tuned = _tuned_rows(os.path.abspath(tuned_config_path())).get(
            tuple(str(k) for k in key), []
        )
        self._cfgs: dict = {}

    # -- work schedule -----------------------------------------------------
    def _pieces(self, rem: int) -> int:
        nb = self.I // 128
        options = [1] + ([2] if nb % 2 == 0 and nb > 2 else []) + [nb]
        for p in options:
            if rem * p >= self.n_cta or p == nb:
                return p
        return nb

    def _schedule(self, xsplit_ok: bool = True) -> _Sched:
        """Every CTA takes E // n_cta whole experts; the leftover experts are cut
        into inter slices dealt one per CTA (their GEMM2 K-slice partials are
        summed by the ReduceScatter push), or, with xsplit_ok, column split."""
        E, C, I = self.E, self.n_cta, self.I
        full, rem = divmod(E, C)
        per = [[] for _ in range(C)]
        self._cols = []
        e = 0
        for c in range(C):
            for _ in range(full):
                per[c].append((e, 0, I, 0))
                e += 1
        self._xsplit = 0
        self._xrem = 0
        self._xw = 0
        if rem:
            p = self._pieces(rem)
            nslice = I // 128
            xok = (
                rem <= XQ_MAX
                and nslice <= XQ_P
                and xsplit_ok
            )
            if xok and rem * p >= C and (rem * p) % C:
                # Many leftover experts whose inter pieces do not deal evenly
                # (dsv4 / kimi3 at TP8: 128 experts x 3 pieces on 256 CTAs --
                # half the CTAs would carry a third of an expert more, and a
                # CTA does not stream faster when others idle). Each leftover
                # expert's GEMM1 is cut into exporting inter slices, dealt
                # round robin, and its GEMM2 into chunk-aligned column slices,
                # dealt chunk by chunk to the least loaded CTA: every CTA ends
                # within a slice of the mean, and the slices finish the
                # chunks in order.
                self._schedule_balanced(per, e, rem, nslice)
                p = 1
            elif xok and rem * p < C:
                # Few leftover experts (glm5: 257 on 256 CTAs): inter pieces
                # would leave a handful of CTAs a quarter expert behind, and a
                # piece's GEMM2 still pays the full-width epilogue. Instead a
                # leftover expert's GEMM1 is cut into 128-column inter slices
                # that export the quantized intermediate, and its GEMM2 into
                # column slices over the whole inter dim, one per CTA. A CTA
                # that takes an inter slice also exports its own expert's
                # GEMM1 and hands that GEMM2 to column slices too, so it ends
                # up with ~5/6 of an expert plus its slice. Column slices go
                # last on their CTAs (every export is long done by then) and
                # count themselves into their chunk's readiness; each CTA's
                # UNIT_SIGNAL unit signals every chunk in order.
                self._xsplit = nslice
                g2 = self.H // 256
                step = math.lcm(gemm2_group_step(self.H, I), gemm2_chunk_groups(self.H, I))
                self._xw = step
                hosts = sorted({(C - 1 - idx) % C for idx in range(rem * nslice)})
                slot = {}
                for j in range(rem):
                    slot[e + j] = j
                for c in hosts:
                    slot[per[c][0][0]] = len(slot)
                if len(slot) > XQ_MAX:
                    raise ValueError("too many column-split experts")
                self._xrem = len(slot)

                def kind(k, j, groups=0):
                    return k | (groups << 8) | (j << XQ_SHIFT)

                # a host exports its own expert first (its column slices run
                # on other CTAs right after their own expert), then its slice
                # of the leftover expert
                for c in hosts:
                    e0 = per[c][0][0]
                    per[c] = [(e0, 0, I, kind(UNIT_G1X, slot[e0]))]
                for idx in range(rem * nslice):
                    j, k = divmod(idx, nslice)
                    per[(C - 1 - idx) % C].append((e + j, k * 128, 128, kind(UNIT_G1X, slot[e + j])))
                # chunk order: every expert's first column slice first
                self._cols = [
                    (ex, g, 0, kind(UNIT_G2COL, j, step))
                    for g in range(0, g2, step)
                    for ex, j in slot.items()
                ]
                # the full expert signals the chunks as its GEMM2 goes; a host
                # (no GEMM2 of its own) on its first unit, right away
                for c in range(C):
                    k = next((k for k in range(len(per[c])) if per[c][k][3] == 0), 0)
                    e0, i0, icnt, kd = per[c][k]
                    per[c][k] = (e0, i0, icnt, kd | UNIT_SIGNAL)
                p = 1
            else:
                width = I // p
                pieces = [(e + j, k * width, width, 0) for j in range(rem) for k in range(p)]
                # a CTA's piece goes first: a column chunk is ready for the
                # ReduceScatter once every CTA's LAST unit produced it, and a full
                # expert's (longer) GEMM2 spreads those chunks over more time
                for idx, piece in enumerate(pieces):
                    per[(C - 1 - idx) % C].insert(0, piece)
        units, recs = [], []
        for lst in per:
            # per CTA: unit range [ub, ue) and its first unit, read at entry
            recs.append([len(units), len(units) + len(lst)] + list(lst[0] if lst else (0, 0, 0, 0)))
            recs[-1] += [0] * (UNIT_REC - len(recs[-1]))
            for unit in lst:
                units.append(list(unit))
        col0 = len(units)
        units += [list(u) for u in self._cols]
        dev = self.device
        return _Sched(
            torch.tensor(units or [[0, 0, 0, 0]], dtype=torch.int32, device=dev),
            torch.tensor(recs, dtype=torch.int32, device=dev),
            p if rem else 0,
            full * C,
            self._xsplit,
            self._xrem,
            self._xw,
            col0,
            len(self._cols),
        )

    def _schedule_balanced(self, per, e, rem, nslice):
        H, I, C = self.H, self.I, len(per)
        g2 = H // 256
        unit = math.lcm(gemm2_group_step(H, I), gemm2_chunk_groups(H, I))
        self._xsplit, self._xrem, self._xw = nslice, rem, unit

        def kind(k, j, groups=0):
            return k | (groups << 8) | (j << XQ_SHIFT)

        # cost model in weight bytes; a unit's fixed cost (route scan, import
        # or export, pipeline refill) as the bytes streamed meanwhile
        full, g1x, grp = 3 * I * H // 2, 128 * H + 32 * 1024, 256 * I // 2
        over = 64 * 1024
        load = [full * len(lst) for lst in per]
        for idx in range(rem * nslice):
            j, k = divmod(idx, nslice)
            c = (C - 1 - idx) % C
            per[c].insert(0, (e + j, k * 128, 128, kind(UNIT_G1X, j)))
            load[c] += g1x
        # GEMM2 of every leftover expert in n chunk-aligned column slices
        # (~4 per CTA), in chunk order. The first rounds are dealt to the
        # least loaded CTA up front; the last ones are claimed at run time
        # (col_claim) by whichever CTAs run out of work first, which absorbs
        # what the byte model misses (XCD speed, import waits). A claim costs
        # a device atomic and two barriers, so only the tail is dynamic.
        nu = g2 // unit
        n = max(1, min(nu, round(4 * C / rem)))
        dyn_rounds = max(1, n // 3)
        cuts = [unit * (nu * t // n) for t in range(n + 1)]
        heap = [(load[c], c) for c in range(C)]
        heapq.heapify(heap)
        for t in range(n - dyn_rounds):
            g0, g1 = cuts[t], cuts[t + 1]
            for j in range(rem):
                lc, c = heapq.heappop(heap)
                per[c].append((e + j, g0, 0, kind(UNIT_G2COL, j, g1 - g0)))
                heapq.heappush(heap, (lc + (g1 - g0) * grp + over, c))
        self._cols = [
            (e + j, cuts[t], 0, kind(UNIT_G2COL, j, cuts[t + 1] - cuts[t]))
            for t in range(n - dyn_rounds, n)
            for j in range(rem)
        ]
        # the last full expert signals the chunks (its route rows are the
        # CTA's last ones before the column slices, which count themselves)
        for c in range(C):
            # a CTA without a full expert (m3: fewer experts than CTAs) counts
            # into every chunk on its first unit
            full_k = [k for k in range(len(per[c])) if per[c][k][3] == 0]
            k = max(full_k) if full_k else 0
            e0, i0, icnt, kd = per[c][k]
            per[c][k] = (e0, i0, icnt, kd | UNIT_SIGNAL)

    def _lds(self, mt: int, dyn: bool, nsk: int, nab: int = 0) -> int:
        """LDS bytes of an instance; nab=0: the A ring depth _nab picks."""
        return fused_tp_consts(
            self.H, self.I, self.K, mt, self.mmax * self.tp, self.agr,
            self.E if dyn else 0, nsk, nab or self._nab(mt, dyn, nsk),
        )["LDS_BYTES"]

    def _nab(self, mt: int, dyn: bool, nsk: int) -> int:
        """A ring buffers: 4, or 3 when that is what lets the row tile fit."""
        big = fused_tp_consts(
            self.H, self.I, self.K, mt, self.mmax * self.tp, self.agr,
            self.E if dyn else 0, nsk, 4,
        )["LDS_BYTES"]
        return 4 if big <= LDS_LIMIT else 3

    def _fit_mt(self, mt: int, dyn: bool, nsk: int) -> int:
        """The largest row tile up to mt whose LDS fits."""
        mt = max(1, min(6, int(mt)))
        while mt > 1 and self._lds(mt, dyn, nsk) > LDS_LIMIT:
            mt -= 1
        return mt

    def default_config(self, m: int) -> LaunchCfg:
        """Heuristic config for m local tokens (no tuned row)."""
        tot = m * self.tp
        # dynamic schedule when the routes can leave experts idle
        dyn = (
            tot <= self.dyn_max
            and self.I // 128 >= 2
            and tot * self.K <= 2 * self.E
        )
        rpe = (tot * self.K + self.E - 1) // self.E
        mt = self._fit_mt((rpe + 15) // 16, dyn, 4)
        nsk = 4
        # bf16 route rows at small batches (the route traffic is small there);
        # LL ReduceScatter packets while the RS is latency, not bandwidth, bound
        return LaunchCfg(mt=mt, nsk=nsk, dyn=dyn, route_fp8=not dyn, xsplit=True,
                         ll=tot <= 256, llr=tot <= 128)

    def config(self, m: int) -> LaunchCfg:
        """The launch config for m local tokens: the tuned CSV row of the
        smallest tuned token count >= m * tp (else the largest), else the
        heuristic; AITER_MEGAMOE_{BLOCK_M,NSK,DYN,ROUTE_FP8,XSPLIT} override."""
        cfg = self._cfgs.get(m)
        if cfg is not None:
            return cfg
        tot = m * self.tp
        cfg = None
        if self._tuned and os.environ.get("AITER_MEGAMOE_TUNED", "1") == "1":
            cfg = next((c for t, c in self._tuned if t >= tot), self._tuned[-1][1])
        if cfg is None:
            cfg = self.default_config(m)
        env = os.environ.get
        over = {}
        if env("AITER_MEGAMOE_BLOCK_M"):
            over["mt"] = int(env("AITER_MEGAMOE_BLOCK_M")) // 16
        if env("AITER_MEGAMOE_NSK"):
            over["nsk"] = int(env("AITER_MEGAMOE_NSK"))
        for name, field in (("DYN", "dyn"), ("ROUTE_FP8", "route_fp8"), ("XSPLIT", "xsplit"),
                            ("LL", "ll"), ("LLR", "llr")):
            v = env(f"AITER_MEGAMOE_{name}", "")
            if v in ("0", "1"):
                over[field] = v == "1"
        if over:
            cfg = LaunchCfg(**{**cfg.__dict__, **over})
        # what this engine can run: dyn needs its partial-row buffers, the
        # tile and ring must fit LDS at this engine's TMAX
        dyn = cfg.dyn and tot <= self.dyn_max and self.I // 128 >= 2
        nsk = cfg.nsk if (self.H // 128) % cfg.nsk == 0 and cfg.nsk % 2 == 0 else 4
        # the static column split's slices are cut for the default ring (its
        # GEMM2 steps GPI column groups at a time, and GPI depends on nsk)
        if not dyn and cfg.xsplit and self._scheds[True].xsplit:
            nsk = 4
        mt = self._fit_mt(cfg.mt, dyn, nsk)
        if self._lds(mt, dyn, nsk) > LDS_LIMIT:
            nsk = 4
            mt = self._fit_mt(cfg.mt, dyn, nsk)
        ll = cfg.ll and self.rs_fp8
        # dyn + LL: the route rows are E4M3 LL packets too
        cfg = LaunchCfg(mt=mt, nsk=nsk, dyn=dyn, route_fp8=cfg.route_fp8,
                        xsplit=cfg.xsplit, ll=ll, llr=cfg.llr and ll)
        self._cfgs[m] = cfg
        return cfg

    def _launcher(self, cfg: LaunchCfg):
        sc = self._scheds[cfg.xsplit]
        # the dynamic schedule plans its own units (no static pieces / splits)
        static = {} if cfg.dyn else dict(
            npieces=sc.npieces, xsplit=sc.xsplit, xrem=sc.xrem, xw=sc.xw
        )
        return compile_fused_tp(
            H=self.H,
            I=self.I,
            TOPK=self.K,
            MT=cfg.mt,
            TMAX=self.mmax * self.tp,
            act=self.activation,
            situ_beta=self.situ[0],
            situ_linear_beta=self.situ[1],
            # (LL route rows -- dyn, or static without pieces -- are E4M3
            # LL packets)
            route_fp8=cfg.route_fp8 or (cfg.ll and cfg.llr and (cfg.dyn or sc.npieces <= 1)),
            rs_fp8=self.rs_fp8,
            agr=self.agr,
            tp=self.tp,
            ar=self.ar,
            timeline=self.tl is not None,
            dyn_e=self.E if cfg.dyn else 0,
            nsk=cfg.nsk,
            nab=self._nab(cfg.mt, cfg.dyn, cfg.nsk),
            ll_rs=cfg.ll,
            ll_route=cfg.ll and cfg.llr,
            **static,
        )

    def forward(
        self, x_local: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ):
        m = int(x_local.shape[0])
        if self.ar:
            if m % self.tp:
                raise ValueError(f"ar_ar: {m} tokens are not a multiple of tp={self.tp}")
            m //= self.tp
        if m > self.mmax:
            raise ValueError(f"{m} local tokens exceed max_local_tokens {self.mmax}")
        x = x_local.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        cfg = self.config(m)
        sc = self._scheds[cfg.xsplit]
        key = (m, x.data_ptr(), ids.data_ptr(), tw.data_ptr(), cfg)
        args = self._args.get(key)
        if args is None:
            peers = [int(b) for b in self.arena.base_ptrs] + [0] * (MAX_TP - self.tp)
            args = (
                self.w1.data_ptr(),
                self.w1s.data_ptr(),
                self.w2.data_ptr(),
                self.w2s.data_ptr(),
                x.data_ptr(),
                ids.data_ptr(),
                tw.data_ptr(),
                self.y.data_ptr(),
                self.routes.data_ptr(),
                self.proutes.data_ptr(),
                self.ctrl.data_ptr(),
                sc.units.data_ptr(),
                sc.recs.data_ptr(),
                *peers,
                self._x.offset,
                self._xs.offset,
                self._ids.offset,
                self._w.offset,
                self._recv.offset,
                self._flag.offset,
                self._pre.offset,
                self._yall.offset,
                self.rank,
                self.tp,
                m,
                self.mmax,
                int(sc.npieces),
                0 if cfg.dyn else int(sc.piece_e0),
                0 if self.tl is None else self.tl.data_ptr(),
                0 if self.xg is None else self.xg.data_ptr(),
                0 if cfg.dyn else int(sc.col0),
                0 if cfg.dyn else int(sc.ncol),
                self.n_cta,
            )
            self._args[key] = (args, (x, ids, tw))
        else:
            args = args[0]
        _run_compiled(self._launcher(cfg), *args, torch.cuda.current_stream())
        if self.ar:
            return self._yall.local[: m * self.tp]
        return self.y[:m]

    __call__ = forward

    def poll_errors(self) -> int:
        """OR of the watchdog codes of every wait that gave up so far (0 when
        healthy); a launch with a nonzero code produced invalid output."""
        return int(self.ctrl[CTRL_ERR].item())

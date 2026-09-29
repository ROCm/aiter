# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel TP MegaMoE (:mod:`.mega_moe_tp_kernel`).

The launch config (:class:`LaunchCfg`) comes from the tuned CSV
``aiter/configs/model_configs/mega_moe_tp_a4w4_tuned.csv`` (by
``op_tests/multigpu_tests/tune_mega_moe_tp.py``; ``AITER_MEGAMOE_TP_CONFIG``
points elsewhere), else from a heuristic.
"""

from __future__ import annotations

import csv
import functools
import heapq
import math
import os
from dataclasses import dataclass

import torch

from ..symmetric_arena import SymmetricArena
from ..tensor_shim import _run_compiled
from .mega_moe_tp_kernel import (
    CTRL_ERR,
    CTRL_INTS,
    FLAG_INTS,
    MAX_TP,
    NCTA_MAX,
    UNIT_G1X,
    UNIT_G2COL,
    UNIT_REC,
    UNIT_SIGNAL,
    XQ_MAX,
    XQ_P,
    XQ_SHIFT,
    compile_mega_moe_tp,
    gemm2_chunk_groups,
    gemm2_group_step,
    mega_moe_tp_consts,
    mega_moe_tp_shape_supported,
)

__all__ = [
    "LaunchCfg",
    "MegaMoeTPEngine",
    "mega_moe_tp_shape_supported",
    "tuned_config_path",
]

LDS_LIMIT = 160 * 1024
DYN_MAX = 256
TUNED_CSV = "mega_moe_tp_a4w4_tuned.csv"
CSV_KEY = (
    "gfx",
    "cu_num",
    "tp",
    "comm_mode",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act",
)
CSV_CFG = ("block_m", "dyn", "route_fp8", "ll", "llr")


@dataclass(frozen=True)
class LaunchCfg:
    """Row tile block_m = 16 * mt; dynamic schedule; E4M3 (else bf16) route
    rows; LL ReduceScatter packets; (llr, with ll) LL route rows plus an LL
    routing AllGather (ag_rs) or LL output all-reduce (ar_ar)."""

    mt: int
    dyn: bool
    route_fp8: bool
    ll: bool = False
    llr: bool = False

    @property
    def block_m(self) -> int:
        return 16 * self.mt


def tuned_config_path() -> str:
    return os.environ.get("AITER_MEGAMOE_TP_CONFIG") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "configs",
        "model_configs",
        TUNED_CSV,
    )


@functools.cache
def _tuned_rows(path: str) -> dict:
    rows: dict = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            cfg = LaunchCfg(
                mt=int(r["block_m"]) // 16,
                dyn=r["dyn"] == "1",
                route_fp8=r["route_fp8"] == "1",
                ll=r["ll"] == "1",
                llr=r["llr"] == "1",
            )
            rows.setdefault(tuple(r[k] for k in CSV_KEY), []).append(
                (int(r["token"]), cfg)
            )
    for v in rows.values():
        v.sort(key=lambda t: t[0])
    return rows


@dataclass
class _Sched:

    units: torch.Tensor
    recs: torch.Tensor
    npieces: int = 0
    piece_e0: int = 0
    xsplit: int = 0
    xrem: int = 0
    xw: int = 0
    col0: int = 0
    ncol: int = 0


def _kind(k, j, groups=0):
    return k | (groups << 8) | (j << XQ_SHIFT)


class MegaMoeTPEngine:

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
        if not mega_moe_tp_shape_supported(model_dim, inter_dim, world_size):
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
        self.w1, self.w1s, self.w2, self.w2s = (
            t.view(torch.uint8) for t in (w1, w1_scale, w2, w2_scale)
        )

        tot = self.mmax * self.tp
        H, K = model_dim, topk
        arena = SymmetricArena(group=group, device=self.device)
        self._x = arena.reserve("x", (tot, H // 2), torch.uint8)
        self._xs = arena.reserve("xs", (tot, H // 32), torch.uint8)
        self._ids = arena.reserve("ids", (tot, 4 * K), torch.int32)
        self._w = arena.reserve("w", (tot, K), torch.float32)
        self._recv = arena.reserve(
            "recv", (self.tp, tot if self.ar else self.mmax, H), torch.bfloat16
        )
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        self._pre = arena.reserve(
            "pre", (self.tp, self.mmax, H) if self.ar else (1,), torch.bfloat16
        )
        self._yall = arena.reserve(
            "yall", (tot, H) if self.ar else (1,), torch.bfloat16
        )
        arena.commit()
        self.arena = arena
        self.ctrl = torch.zeros(CTRL_INTS, dtype=torch.int32, device=self.device)
        self.routes = torch.empty(
            (tot * topk + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.y = torch.empty((self.mmax, H), dtype=torch.bfloat16, device=self.device)

        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        if self.n_cta > NCTA_MAX:
            raise ValueError(f"at most {NCTA_MAX} CTAs (flag layout)")
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        self.agr = max(1, -(-self.mmax // self.n_cta))
        self.sched = self._schedule()
        self.dyn_max = min(tot, DYN_MAX)
        extra = (
            max(
                max(self.sched.npieces - 1, 0) * tot,
                (inter_dim // 128 - 1) * self.dyn_max,
            )
            * topk
        )
        if extra * H * 2 >= 1 << 31:
            raise ValueError("split-expert partial rows exceed 32-bit buffer offsets")
        self.proutes = torch.empty(
            (extra + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.xg = (
            torch.empty(
                tot * topk * (inter_dim // 2 + inter_dim // 32),
                dtype=torch.uint8,
                device=self.device,
            )
            if self.sched.xsplit
            else None
        )
        key = (
            self.gfx,
            self.n_cta,
            self.tp,
            comm_mode,
            H,
            inter_dim,
            experts,
            topk,
            activation,
        )
        self._tuned = _tuned_rows(os.path.abspath(tuned_config_path())).get(
            tuple(str(k) for k in key), []
        )
        self._cfgs: dict = {}
        self._launchers: dict = {}
        self._args = (None, None)

    def _pieces(self, rem: int) -> int:
        nb = self.I // 128
        for p in [1] + ([2] if nb % 2 == 0 and nb > 2 else []):
            if rem * p >= self.n_cta:
                return p
        return nb

    def _schedule(self) -> _Sched:
        E, C, I = self.E, self.n_cta, self.I
        full, rem = divmod(E, C)
        per = [[(c * full + k, 0, I, 0) for k in range(full)] for c in range(C)]
        e = full * C
        sc = {"piece_e0": e}
        cols = []
        if rem:
            p = self._pieces(rem)
            nslice = I // 128
            xok = rem <= XQ_MAX and nslice <= XQ_P
            if xok and rem * p >= C and (rem * p) % C:
                cols = self._schedule_balanced(per, e, rem, nslice)
                sc.update(xsplit=nslice, xrem=rem)
            elif xok and rem * p < C:
                cols = self._schedule_few(per, e, rem, nslice, sc)
            else:
                width = I // p
                pieces = [
                    (e + j, k * width, width, 0) for j in range(rem) for k in range(p)
                ]
                for idx, piece in enumerate(pieces):
                    per[(C - 1 - idx) % C].insert(0, piece)
                sc["npieces"] = p
        if cols:
            sc["xw"] = math.lcm(
                gemm2_group_step(self.H, I), gemm2_chunk_groups(self.H, I)
            )
        units, recs = [], []
        for lst in per:
            recs.append(
                [len(units), len(units) + len(lst)]
                + list(lst[0] if lst else (0, 0, 0, 0))
            )
            recs[-1] += [0] * (UNIT_REC - len(recs[-1]))
            units += [list(u) for u in lst]
        sc.update(col0=len(units), ncol=len(cols))
        units += [list(u) for u in cols]
        dev = self.device
        return _Sched(
            torch.tensor(units or [[0, 0, 0, 0]], dtype=torch.int32, device=dev),
            torch.tensor(recs, dtype=torch.int32, device=dev),
            **sc,
        )

    def _schedule_few(self, per, e, rem, nslice, sc):
        C, I = len(per), self.I
        step = math.lcm(gemm2_group_step(self.H, I), gemm2_chunk_groups(self.H, I))
        hosts = sorted({(C - 1 - idx) % C for idx in range(rem * nslice)})
        slot = {e + j: j for j in range(rem)}
        for c in hosts:
            slot[per[c][0][0]] = len(slot)
        sc.update(xsplit=nslice, xrem=len(slot))
        for c in hosts:
            e0 = per[c][0][0]
            per[c] = [(e0, 0, I, _kind(UNIT_G1X, slot[e0]))]
        for idx in range(rem * nslice):
            j, k = divmod(idx, nslice)
            per[(C - 1 - idx) % C].append(
                (e + j, k * 128, 128, _kind(UNIT_G1X, slot[e + j]))
            )
        for c in range(C):
            k = next((k for k in range(len(per[c])) if per[c][k][3] == 0), 0)
            per[c][k] = per[c][k][:3] + (per[c][k][3] | UNIT_SIGNAL,)
        return [
            (ex, g, 0, _kind(UNIT_G2COL, j, step))
            for g in range(0, self.H // 256, step)
            for ex, j in slot.items()
        ]

    def _schedule_balanced(self, per, e, rem, nslice):
        H, I, C = self.H, self.I, len(per)
        g2 = H // 256
        unit = math.lcm(gemm2_group_step(H, I), gemm2_chunk_groups(H, I))
        full, g1x, grp, over = (
            3 * I * H // 2,
            128 * H + 32 * 1024,
            256 * I // 2,
            64 * 1024,
        )
        load = [full * len(lst) for lst in per]
        for idx in range(rem * nslice):
            j, k = divmod(idx, nslice)
            c = (C - 1 - idx) % C
            per[c].insert(0, (e + j, k * 128, 128, _kind(UNIT_G1X, j)))
            load[c] += g1x
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
                per[c].append((e + j, g0, 0, _kind(UNIT_G2COL, j, g1 - g0)))
                heapq.heappush(heap, (lc + (g1 - g0) * grp + over, c))
        for c in range(C):
            full_k = [k for k in range(len(per[c])) if per[c][k][3] == 0]
            k = max(full_k) if full_k else 0
            per[c][k] = per[c][k][:3] + (per[c][k][3] | UNIT_SIGNAL,)
        return [
            (e + j, cuts[t], 0, _kind(UNIT_G2COL, j, cuts[t + 1] - cuts[t]))
            for t in range(n - dyn_rounds, n)
            for j in range(rem)
        ]

    def _consts(self, mt: int, dyn: bool, nab: int = 4) -> dict:
        return mega_moe_tp_consts(
            self.H, self.I, mt, self.mmax * self.tp, self.agr, self.E if dyn else 0, nab
        )

    def _nab(self, mt: int, dyn: bool) -> int:
        return 4 if self._consts(mt, dyn)["LDS_BYTES"] <= LDS_LIMIT else 3

    def _lds(self, mt: int, dyn: bool) -> int:
        return self._consts(mt, dyn, self._nab(mt, dyn))["LDS_BYTES"]

    def _fit_mt(self, mt: int, dyn: bool) -> int:
        mt = max(1, min(6, int(mt)))
        while mt > 1 and self._lds(mt, dyn) > LDS_LIMIT:
            mt -= 1
        return mt

    def default_config(self, m: int) -> LaunchCfg:
        tot = m * self.tp
        dyn = tot <= self.dyn_max and self.I // 128 >= 2 and tot * self.K <= 2 * self.E
        rpe = (tot * self.K + self.E - 1) // self.E
        return LaunchCfg(
            mt=self._fit_mt((rpe + 15) // 16, dyn),
            dyn=dyn,
            route_fp8=not dyn,
            ll=tot <= 256,
            llr=tot <= 128,
        )

    def config(self, m: int) -> LaunchCfg:
        """The tuned row of the smallest tuned token count >= m * tp (else the
        largest), else the heuristic; clamped to what this engine can run."""
        cfg = self._cfgs.get(m)
        if cfg is None:
            tot = m * self.tp
            if self._tuned:
                cfg = next((c for t, c in self._tuned if t >= tot), self._tuned[-1][1])
            else:
                cfg = self.default_config(m)
            dyn = cfg.dyn and tot <= self.dyn_max and self.I // 128 >= 2
            cfg = LaunchCfg(
                self._fit_mt(cfg.mt, dyn),
                dyn,
                cfg.route_fp8,
                cfg.ll,
                cfg.ll and cfg.llr,
            )
            self._cfgs[m] = cfg
        return cfg

    def _launcher(self, cfg: LaunchCfg):
        fn = self._launchers.get(cfg)
        if fn is None:
            sc = self.sched
            static = (
                {}
                if cfg.dyn
                else {
                    "npieces": sc.npieces,
                    "xsplit": sc.xsplit,
                    "xrem": sc.xrem,
                    "xw": sc.xw,
                }
            )
            fn = compile_mega_moe_tp(
                H=self.H,
                I=self.I,
                TOPK=self.K,
                MT=cfg.mt,
                TMAX=self.mmax * self.tp,
                act=self.activation,
                situ_beta=self.situ[0],
                situ_linear_beta=self.situ[1],
                route_fp8=cfg.route_fp8
                or (cfg.ll and cfg.llr and (cfg.dyn or sc.npieces <= 1)),
                agr=self.agr,
                tp=self.tp,
                ar=self.ar,
                dyn_e=self.E if cfg.dyn else 0,
                nab=self._nab(cfg.mt, cfg.dyn),
                ll_rs=cfg.ll,
                ll_route=cfg.ll and cfg.llr,
                **static,
            )
            self._launchers[cfg] = fn
        return fn

    def forward(
        self, x_local: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ):
        m = int(x_local.shape[0])
        if self.ar:
            if m % self.tp:
                raise ValueError(
                    f"ar_ar: {m} tokens are not a multiple of tp={self.tp}"
                )
            m //= self.tp
        if m > self.mmax:
            raise ValueError(f"{m} local tokens exceed max_local_tokens {self.mmax}")
        x = x_local.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        cfg = self.config(m)
        key = (m, x.data_ptr(), ids.data_ptr(), tw.data_ptr(), cfg)
        if self._args[0] != key:
            sc = self.sched
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
                0 if cfg.dyn else sc.piece_e0,
                0 if self.xg is None else self.xg.data_ptr(),
                0 if cfg.dyn else sc.col0,
                0 if cfg.dyn else sc.ncol,
                self.n_cta,
            )
            # the inputs stay referenced while their pointers are cached
            self._args = (key, (args, x, ids, tw))
        _run_compiled(
            self._launcher(cfg), *self._args[1][0], torch.cuda.current_stream()
        )
        if self.ar:
            return self._yall.local[: m * self.tp]
        return self.y[:m]

    __call__ = forward

    def poll_errors(self) -> int:
        """OR of the watchdog codes of every wait that gave up (0 when healthy);
        a launch with a nonzero code produced invalid output."""
        return int(self.ctrl[CTRL_ERR].item())

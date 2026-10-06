# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel TP MegaMoE (:mod:`.mega_moe_tp_kernel`).

The launch config (:class:`LaunchCfg`) comes from the tuned CSV
``aiter/configs/mega_moe_tp_a4w4_tuned.csv`` (by
``op_tests/tuners/tune_mega_moe_tp.py``; ``AITER_CONFIG_MEGAMOE_TP`` points
elsewhere, as for the other AITER_CONFIG_* tables), else from a heuristic.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import heapq
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..symmetric_arena import SymmetricArena
from ..tensor_shim import _preload_compiled, _run_compiled
from .mega_moe_tp_kernel import (
    CTRL_ERR,
    CTRL_INTS,
    DYN_MAX,
    ERR_CHUNK,
    ERR_COMM,
    ERR_FLAG,
    ERR_META,
    ERR_YAG,
    FLAG_INTS,
    MAX_TP,
    NCTA_MAX,
    TLU_INTS,
    TN_MAX,
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
# LB: up to this many tokens the A loader wave prefetches the GEMM2 units and
# the GEMM1 split is picked per batch at run time
LB_PF_MAX = 1024
COMM_MODES = ("ag_rs", "rs", "ar", "ar_ar")
_TUNED_LIKE = {"ar": "ar_ar", "rs": "ag_rs"}
_ERR_NAMES = (
    (ERR_FLAG, "flag"),
    (ERR_META, "routing"),
    (ERR_CHUNK, "input chunk"),
    (ERR_COMM, "reduce"),
    (ERR_YAG, "output gather"),
)
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
CSV_CFG = ("block_m", "nsk", "npp", "xb", "dyn", "route_fp8", "ll", "llr")


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
    nsk: int = 4
    npp: int = 1
    xb: int = 0
    ag8: bool = False  # ar / ar_ar: MXFP8 all-gather rows (ar_gather="auto")
    lb: bool = False  # dynamic: two-phase large-batch schedule (no partial sums)
    tn: int = 0  # ag_rs: the next layer's residual add + norm + FP8 gather fused (1)
    pf: bool = False  # lb: the A loader wave prefetches the GEMM2 units
    lp: int = 0  # lb: GEMM1 inter pieces per row chunk (0: picked per batch)

    @property
    def block_m(self) -> int:
        return 16 * self.mt


def tuned_config_path() -> str:
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_MEGAMOE_TP_FILE


@functools.cache
def _tuned_rows(path: str) -> dict:
    rows: dict = {}
    if not os.path.exists(path):
        return rows

    def num(r, k, default=0):
        # merged tables (pandas) can carry "4.0" / "" for integer columns
        v = (r.get(k) or "").strip()
        return int(float(v)) if v and v.lower() != "nan" else default

    def key(v):
        try:
            return str(int(float(v)))
        except ValueError:
            return v

    with open(path) as f:
        for r in csv.DictReader(f):
            cfg = LaunchCfg(
                mt=num(r, "block_m") // 16,
                dyn=num(r, "dyn") == 1,
                route_fp8=num(r, "route_fp8") == 1,
                ll=num(r, "ll") == 1,
                llr=num(r, "llr") == 1,
                nsk=num(r, "nsk", 4),
                npp=num(r, "npp", 1),
                xb=num(r, "xb"),
            )
            rows.setdefault(tuple(key(r[k]) for k in CSV_KEY), []).append(
                (num(r, "token"), cfg)
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
    xl_e0: int = 0
    xl_s0: int = 0


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
        swiglu_limit: float | None = None,
        comm_mode: str = "ag_rs",
        comm_dtype: str = "fp8",
        ar_gather: str = "auto",
        schedule: str = "tuned",
        act_dtype: str = "fp4",
        group=None,
        device: torch.device | None = None,
    ):
        if not mega_moe_tp_shape_supported(model_dim, inter_dim, world_size):
            raise ValueError(
                f"fused TP MegaMoE does not tile h{model_dim} i{inter_dim} tp{world_size}"
            )
        if comm_mode not in COMM_MODES:
            raise ValueError(f"unknown comm_mode {comm_mode!r}")
        if comm_dtype not in ("fp8", "bf16"):
            raise ValueError(f"unknown comm_dtype {comm_dtype!r}")
        self.comm_bf16 = comm_dtype == "bf16"
        if ar_gather not in ("bf16", "fp8", "auto"):
            raise ValueError(f"unknown ar_gather {ar_gather!r}")
        self.ag_fp8 = ar_gather == "fp8"
        # auto: MXFP8 gathered rows from AG8_MIN global tokens up, where the
        # second all-reduce hop is link bound (as comm-fused MoE's reduced payload)
        self.ag_auto = (
            int(os.environ.get("AITER_MEGAMOE_TP_AG8_MIN", "512"))
            if ar_gather == "auto"
            else 0
        )
        if not 1 <= topk <= experts or max_local_tokens < 1:
            raise ValueError(
                f"need 1 <= topk ({topk}) <= experts ({experts}), max_local_tokens >= 1"
            )
        self.mode = comm_mode
        self.check = os.environ.get("AITER_MEGAMOE_TP_CHECK", "0") == "1"
        self.dx = os.environ.get("AITER_MEGAMOE_TP_DX", "1") == "1"
        # "tuned": the tuned table (balanced routing). "dynamic": the dynamic
        # schedule with row chunks for every batch it can run, which is what
        # model routing (idle and hot experts) wants.
        schedule = os.environ.get("AITER_MEGAMOE_TP_SCHEDULE", schedule)
        if schedule not in ("tuned", "dynamic"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.schedule = schedule
        # dynamic schedule: split experts routed by more than one row tile into
        # row chunks run by different CTAs (real routing has hot experts)
        self.chunk = (
            os.environ.get("AITER_MEGAMOE_TP_CHUNK", "1" if schedule == "dynamic" else "0")
            == "1"
        )
        # activations (input and GEMM2 intermediate) as MXFP4 or MXFP8 (a8);
        # the MXFP8 layout exists on the dynamic schedule only
        act_dtype = os.environ.get("AITER_MEGAMOE_TP_ACT", act_dtype)
        if act_dtype not in ("fp4", "fp8"):
            raise ValueError(f"unknown act_dtype {act_dtype!r}")
        self.a8 = act_dtype == "fp8"
        if self.a8 and schedule != "dynamic":
            raise ValueError('act_dtype="fp8" needs schedule="dynamic"')
        # dynamic schedule: global token counts from which the large-batch
        # (two-phase) schedule runs (0: never)
        self.lb_min = (
            int(os.environ.get("AITER_MEGAMOE_TP_LB_MIN", "0"))
            if schedule == "dynamic" and not self.a8
            else 0
        )
        self.lb_mt = int(os.environ.get("AITER_MEGAMOE_TP_LB_MT", "3"))
        # smaller row tiles up to lb_small_max global tokens (rows per expert
        # are few there: less padded work per unit)
        self.lb_mt_small = int(os.environ.get("AITER_MEGAMOE_TP_LB_MT_SMALL", "3"))
        self.lb_small_max = int(os.environ.get("AITER_MEGAMOE_TP_LB_SMALL_MAX", "512"))
        # GEMM1 column blocks per pass over A (up to lb_small_max: 2)
        self.lb_npp = int(os.environ.get("AITER_MEGAMOE_TP_LB_NPP", "1"))
        # output column chunks per GEMM2 column group
        self.lb_q = int(os.environ.get("AITER_MEGAMOE_TP_LB_Q", "1"))
        self.ar = comm_mode in ("ar", "ar_ar")
        self.xrep = comm_mode in ("ar", "rs")
        # inputs and routing of every token on every rank
        self.rrep = self.ar or self.xrep
        self.rank, self.tp = int(rank), int(world_size)
        self.H, self.I, self.E, self.K = model_dim, inter_dim, experts, topk
        self.mmax = int(max_local_tokens)
        device = torch.device(device if device is not None else "cuda")
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        self.activation = activation
        self.situ = (float(situ_beta), float(situ_linear_beta))
        if swiglu_limit is None:
            swiglu_limit = 7.0 if activation == "swiglu" else float("inf")
        self.swiglu_limit = float(swiglu_limit)
        self._bind_weights(w1, w1_scale, w2, w2_scale)

        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        if self.n_cta > NCTA_MAX:
            raise ValueError(f"at most {NCTA_MAX} CTAs (flag layout)")
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        if self.gfx != "gfx950":
            raise ValueError(f"fused TP MegaMoE needs gfx950, not {self.gfx}")
        self.agr = max(1, -(-self.mmax // self.n_cta))
        # (the static schedule runs batches above the dynamic one's DYN_MAX
        # tokens and below lb_min: none when LB takes over right there)
        static = not (self.schedule == "dynamic" and self.lb_min and self.lb_min <= DYN_MAX + 1)
        if static and self._lds(1, False) > LDS_LIMIT:
            ok = self.mmax
            while ok > 1 and self._lds(1, False, mmax=ok) > LDS_LIMIT:
                ok -= max(1, ok // 64)
            raise ValueError(
                f"max_local_tokens={self.mmax} does not fit in LDS for h{model_dim} "
                f"i{inter_dim} e{experts} k{topk} tp{world_size}: at most {ok}"
            )

        tot = self.mmax * self.tp
        H, K = model_dim, topk
        arena = SymmetricArena(group=group, device=self.device)
        self._x = arena.reserve("x", (tot, H if self.a8 else H // 2), torch.uint8)
        self._xs = arena.reserve("xs", (tot, H // 32), torch.uint8)
        self._ids = arena.reserve("ids", (2, tot, 4 * K), torch.int32)
        self._w = arena.reserve("w", (2, tot, K), torch.float32)
        self._recv = arena.reserve(
            "recv", (self.tp, tot if self.ar else self.mmax, H), torch.bfloat16
        )
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        self._pre = arena.reserve(
            "pre",
            (self.tp, self.mmax, H) if self.ar and not self.xrep else (1,),
            torch.bfloat16,
        )
        self._yall = arena.reserve(
            "yall", (tot, H) if self.ar else (1,), torch.bfloat16
        )
        # tail (tn, ag_rs): every rank's FP8 rows of the next layer's input and
        # their per-token scales, gathered by the kernel
        # (FP8 rows [tot, H] + scales)
        tn_ok = comm_mode == "ag_rs"
        self._qall = arena.reserve("qall", (tot * H,) if tn_ok else (1,), torch.uint8)
        self._sall = arena.reserve("sall", (tot,) if tn_ok else (1,), torch.float32)
        self.tn_eps = 1e-6
        self.tn_mode = 1
        arena.commit()
        self.arena = arena
        self.ctrl = torch.zeros(CTRL_INTS + 256 * 16 + 128 + TLU_INTS, dtype=torch.int32, device=self.device)  # [tl]
        self.routes = torch.zeros(
            (tot * topk + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.y = torch.empty(
            (tot if self.ar else self.mmax, H), dtype=torch.bfloat16, device=self.device
        )

        self._scheds: dict = {}
        self.sched = self._sched(0)
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
        self.proutes = torch.zeros(
            (extra + 1, H), dtype=torch.bfloat16, device=self.device
        )
        # exported intermediate rows and scales (+ the LB route list)
        self.xg = (
            torch.empty(
                tot * topk * (inter_dim // 2 + inter_dim // 32 + 8),
                dtype=torch.uint8,
                device=self.device,
            )
            if self.sched.xsplit or self.dx or self.lb_min
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
        rows = _tuned_rows(os.path.abspath(tuned_config_path()))
        self._tuned = rows.get(tuple(str(k) for k in key), [])
        if not self._tuned and comm_mode in _TUNED_LIKE:
            # same output collective and schedule: the closest tuned mode
            key = key[:3] + (_TUNED_LIKE[comm_mode],) + key[4:]
            self._tuned = rows.get(tuple(str(k) for k in key), [])
        self._cfgs: dict = {}
        self._launchers: dict = {}
        self._armed: set = set()
        self._args = (None, None)

    def _bind_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        w = tuple(t.view(torch.uint8) for t in (w1, w1_scale, w2, w2_scale))
        H, I, E = self.H, self.I, self.E
        for name, t, n in (
            ("w1", w[0], E * 2 * I * H // 2),
            ("w1_scale", w[1], E * 2 * I * H // 32),
            ("w2", w[2], E * H * I // 2),
            ("w2_scale", w[3], E * H * I // 32),
        ):
            if t.device != self.device or not t.is_contiguous() or t.numel() < n:
                raise ValueError(
                    f"{name}: need a contiguous tensor of >= {n} bytes on {self.device}"
                )
        self.w1, self.w1s, self.w2, self.w2s = w

    def set_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        """Run the next forwards on other weights of this shape; launch configs
        stay armed."""
        self._bind_weights(w1, w1_scale, w2, w2_scale)
        self._args = (None, None)

    def _pieces(self, rem: int) -> int:
        nb = self.I // 128
        for p in [1] + ([2] if nb % 2 == 0 and nb > 2 else []):
            if rem * p >= self.n_cta:
                return p
        return nb

    def _sched(self, xb: int) -> _Sched:
        sc = self._scheds.get(xb)
        if sc is None:
            sc = self._scheds[xb] = self._schedule(xb)
        return sc

    def _schedule(self, xb: int = 0) -> _Sched:
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
            extra = 2 * rem - C
            pairs = (
                xok
                and full == 0
                and nslice % 2 == 0
                and 0 < extra < C
                and extra % 2 == 0
                and (extra // 2) * nslice % 2 == 0
            )
            if pairs:
                cols = self._schedule_pairs(per, rem, nslice, sc, xb)
            elif xok and rem * p >= C and (rem * p) % C:
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

    def _schedule_pairs(self, per, E, nslice, sc, xb=0):
        C, I, H = len(per), self.I, self.H
        g2 = H // 256
        step = math.lcm(gemm2_group_step(H, I), gemm2_chunk_groups(H, I))
        nx = (2 * E - C) // 2
        nh = nx * nslice
        hosts = range(C - nh, C)
        half = nslice // 2
        for c in range(C):
            j, h = divmod(c, 2)
            per[c] = [(j, h * half * 128, half * 128, _kind(UNIT_G1X, j))]
            if c not in hosts:
                lo = 0 if h == 0 else g2 // 2 + xb
                hi = g2 // 2 + xb if h == 0 else g2
                per[c].append((j, lo, 0, _kind(UNIT_G2COL, j, hi - lo)))
        for idx, c in enumerate(hosts):
            j, k = divmod(idx, nslice)
            per[c].append((E - nx + j, k * 128, 128, _kind(UNIT_G1X, E - nx + j)))
        for c in range(C):
            per[c][0] = per[c][0][:3] + (per[c][0][3] | UNIT_SIGNAL,)
        cols = []
        late0 = (C - nh) // 2
        for g in range(0, g2, step):
            for c in hosts:
                j, h = divmod(c, 2)
                if h * (g2 // 2) <= g < (h + 1) * (g2 // 2):
                    cols.append((j, g, 0, _kind(UNIT_G2COL, j, step)))
            for j in range(E - nx, E):
                cols.append((j, g, 0, _kind(UNIT_G2COL, j, step)))
        sc.update(xsplit=nslice, xrem=E, xl_e0=late0, xl_s0=late0, piece_e0=late0)
        return cols

    def _schedule_few(self, per, e, rem, nslice, sc):
        C, I = len(per), self.I
        step = math.lcm(gemm2_group_step(self.H, I), gemm2_chunk_groups(self.H, I))
        hosts = sorted({(C - 1 - idx) % C for idx in range(rem * nslice)})
        slot = {e + j: j for j in range(rem)}
        for c in hosts:
            slot[per[c][0][0]] = len(slot)
        sc.update(xsplit=nslice, xrem=len(slot))
        if sorted(slot) == list(range(min(slot), self.E)):
            sc["xl_e0"] = min(slot)
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

    def _chunks(
        self, mt: int, dyn: bool, mmax: int = 0, lb: bool = False
    ) -> tuple[int, int]:
        """(rows per chunk, chunk table entries) of a dynamic schedule."""
        if not (dyn and (self.chunk or lb)):
            return 0, 0
        rch = 16 * mt
        tot = (mmax or self.mmax) * self.tp
        if not lb:
            tot = min(tot, DYN_MAX)
        return rch, self.E + -(-tot * self.K // rch) + 1

    def _consts(
        self,
        mt: int,
        dyn: bool,
        nab: int = 4,
        nsk: int = 4,
        mmax: int = 0,
        lb: bool = False,
    ) -> dict:
        mmax = mmax or self.mmax
        return mega_moe_tp_consts(
            self.H,
            self.I,
            mt,
            mmax * self.tp,
            max(1, -(-mmax // self.n_cta)),
            self.E if dyn else 0,
            nab,
            nsk,
            self._chunks(mt, dyn, mmax, lb)[1],
            self.a8,
            lb,
        )

    def _nab(
        self, mt: int, dyn: bool, nsk: int = 4, mmax: int = 0, lb: bool = False
    ) -> int:
        lds = self._consts(mt, dyn, 4, nsk, mmax, lb)["LDS_BYTES"]
        if lb:
            return 3
        return 4 if lds <= LDS_LIMIT else 3

    def _lds(
        self, mt: int, dyn: bool, nsk: int = 4, mmax: int = 0, lb: bool = False
    ) -> int:
        nab = self._nab(mt, dyn, nsk, mmax, lb)
        return self._consts(mt, dyn, nab, nsk, mmax, lb)["LDS_BYTES"]

    def _fit_mt(self, mt: int, dyn: bool, nsk: int = 4, lb: bool = False) -> int:
        mt = max(1, min(6, int(mt)))
        while mt > 1 and self._lds(mt, dyn, nsk, lb=lb) > LDS_LIMIT:
            mt -= 1
        return mt

    def _fit_npp(self, npp: int, nsk: int) -> int:
        ok = npp >= 1 and nsk % (2 * npp) == 0 and (self.H // 128 * npp) % nsk == 0
        return npp if ok else 1

    def _fit_nsk(self, mt: int, dyn: bool, nsk: int) -> int:
        nsk = nsk if nsk in (4, 6, 8) and (self.H // 128) % nsk == 0 else 4
        while nsk > 4 and self._lds(mt, dyn, nsk) > LDS_LIMIT:
            nsk -= 2
        return nsk

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
            lb = bool(self.lb_min) and tot >= self.lb_min and self.I // 128 >= 2
            if lb:
                cfg = LaunchCfg(
                    mt=self.lb_mt_small if tot <= self.lb_small_max else self.lb_mt,
                    dyn=True,
                    route_fp8=False,
                    npp=2 if tot <= self.lb_small_max else self.lb_npp,
                    lb=True,
                    # the GEMM2 units prefetched by the A loader wave
                    pf=tot <= LB_PF_MAX,
                    # GEMM1 split: the fewest unit rounds at small batches;
                    # above, a fixed split pipelines GEMM2 better
                    lp=0 if tot <= LB_PF_MAX else 3,
                )
            elif self.schedule == "dynamic" and tot <= self.dyn_max:
                cfg = LaunchCfg(
                    mt=1 if tot <= 32 else 2 if tot <= 64 else 3,
                    dyn=True,
                    route_fp8=False,
                    ll=tot <= 256,
                    llr=tot <= 128,
                )
            elif self._tuned:
                cfg = next((c for t, c in self._tuned if t >= tot), self._tuned[-1][1])
            else:
                cfg = self.default_config(m)
            dyn = cfg.dyn and (lb or tot <= self.dyn_max) and self.I // 128 >= 2
            dyn = dyn and self._lds(1, True, lb=lb) <= LDS_LIMIT
            lb = lb and dyn
            if self.a8 and not dyn:
                raise ValueError(
                    f'act_dtype="fp8": {tot} tokens are beyond the dynamic schedule'
                )
            mt = self._fit_mt(cfg.mt, dyn, lb=lb)
            nsk = 4 if lb else self._fit_nsk(mt, dyn, cfg.nsk)
            fp8 = not self.comm_bf16
            cfg = LaunchCfg(
                mt,
                dyn,
                cfg.route_fp8 and fp8,
                cfg.ll and fp8,
                cfg.ll and cfg.llr and fp8,
                nsk,
                self._fit_npp(cfg.npp, nsk),
                cfg.xb if cfg.xb % 2 == 0 and 0 <= cfg.xb < self.H // 512 else 0,
                bool(self.ag_auto) and tot >= self.ag_auto,
                lb,
                0,
                lb and cfg.pf,
                cfg.lp if lb and self.I // 128 % max(cfg.lp, 1) == 0 else 0,
            )
            self._cfgs[m] = cfg
        return cfg

    def _cfg_sched(self, cfg: LaunchCfg) -> _Sched:
        return self._sched(0 if cfg.dyn else cfg.xb)

    def _arll(self, cfg: LaunchCfg) -> bool:
        sc = self._cfg_sched(cfg)
        return self.ar and cfg.ll and cfg.llr and (cfg.dyn or sc.npieces <= 1)

    def _ag8(self, cfg: LaunchCfg) -> bool:
        return self.ar and (self.ag_fp8 or cfg.ag8) and not self._arll(cfg)

    def _xl(self, cfg: LaunchCfg) -> bool:
        return not cfg.dyn and not cfg.ll and self._cfg_sched(cfg).xl_e0 > 0

    def _launcher(self, cfg: LaunchCfg):
        fn = self._launchers.get(cfg)
        if fn is None:
            sc = self._cfg_sched(cfg)
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
                E=self.E,
                act=self.activation,
                situ_beta=self.situ[0],
                situ_linear_beta=self.situ[1],
                swiglu_limit=self.swiglu_limit,
                route_fp8=cfg.route_fp8
                or (cfg.ll and cfg.llr and (cfg.dyn or sc.npieces <= 1)),
                agr=self.agr,
                tp=self.tp,
                ar=self.ar,
                dyn_e=self.E if cfg.dyn else 0,
                nab=self._nab(cfg.mt, cfg.dyn, cfg.nsk, lb=cfg.lb),
                nsk=cfg.nsk,
                npp=cfg.npp,
                ll_rs=cfg.ll,
                ll_route=cfg.ll and cfg.llr,
                xl=self._xl(cfg),
                xl_s0=sc.xl_s0 if self._xl(cfg) else 0,
                comm_bf16=self.comm_bf16,
                xrep=self.xrep,
                ag8=self._ag8(cfg),
                dx=self.dx and cfg.dyn and cfg.ll and cfg.llr and not self.a8,
                rch=self._chunks(cfg.mt, cfg.dyn, lb=cfg.lb)[0],
                nch=self._chunks(cfg.mt, cfg.dyn, lb=cfg.lb)[1],
                vb=self.schedule != "dynamic",
                a8=self.a8,
                lb=cfg.lb,
                lbq=self.lb_q if cfg.lb else 1,
                tn=cfg.tn,
                tn_eps=self.tn_eps,
                lbpf=cfg.pf,
                lbp=cfg.lp,
                **static,
            )
            self._launchers[cfg] = fn
        return fn

    def _local_tokens(self, x, topk_weights, topk_ids) -> int:
        rows = int(x.shape[0]) if x.dim() == 2 else -1
        for name, t in (
            ("x", x),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
        ):
            if t.device != self.device:
                raise ValueError(f"{name} is on {t.device}, the layer on {self.device}")
        if x.dim() != 2 or x.shape[1] != self.H or x.dtype != torch.bfloat16:
            raise ValueError(
                f"x: need bf16 [tokens, {self.H}], got {x.dtype} {tuple(x.shape)}"
            )
        want = (rows, self.K)
        if tuple(topk_ids.shape) != want or tuple(topk_weights.shape) != want:
            raise ValueError(
                f"topk_ids / topk_weights: need {list(want)}, got "
                f"{list(topk_ids.shape)} / {list(topk_weights.shape)}"
            )
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"topk_ids: need int32 (or int64), got {topk_ids.dtype}")
        if not topk_weights.is_floating_point():
            raise ValueError(f"topk_weights: need float32, got {topk_weights.dtype}")
        m = rows
        if self.rrep:
            if m % self.tp:
                raise ValueError(
                    f"{self.mode}: {m} tokens are not a multiple of tp={self.tp}"
                )
            m //= self.tp
        if m > self.mmax:
            raise ValueError(f"{m} local tokens exceed max_local_tokens {self.mmax}")
        if (
            self.check
            and dist.is_initialized()
            and self.arena.ipc
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._check_replicas(m, x, topk_weights, topk_ids)
        return m

    def _check_replicas(self, m, x, topk_weights, topk_ids) -> None:
        """Debug: the inputs every rank must agree on (one collective per call)."""
        if self.rrep:
            # cheap fingerprints of the replicated inputs (and routing)
            fp = [
                float(t.double().sum())
                for t in (x, topk_weights, topk_ids, (x.float() * x.float()).sum(1))
            ]
        else:
            fp = []
        got = [None] * self.tp
        dist.all_gather_object(got, (m, fp), group=self.arena.group)
        if len({g[0] for g in got}) > 1:
            raise ValueError(
                f"{self.mode}: local token counts differ across ranks: "
                f"{[g[0] for g in got]}"
            )
        if self.rrep and any(g[1] != fp for g in got):
            raise ValueError(
                f"{self.mode}: x / topk_ids / topk_weights must be the same on every "
                "rank (replicated input)"
            )

    def prepare(self, local_tokens, tail: bool = False) -> None:
        """Compile and arm the kernel for these local token counts (collective:
        every rank, same list). forward() does this on first use of a launch
        config, which must not happen inside CUDA graph capture."""
        for m in local_tokens:
            if 0 < m <= self.mmax:
                self._arm(self.config(int(m)), int(m))
                if tail and self.tail_ok(int(m)):
                    self._arm(dataclasses.replace(self.config(int(m)), tn=self.tn_mode), int(m))

    def tail_ok(self, m: int) -> bool:
        """forward(tail=...) runs for m local tokens (ag_rs, the LB schedule)."""
        return self.mode == "ag_rs" and 0 < m <= min(TN_MAX, self.mmax) and self.config(m).lb

    def _arm(self, cfg: LaunchCfg, m: int = 0) -> None:
        if cfg in self._armed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"MegaMoeTP: launch config {cfg} first used inside CUDA graph "
                "capture; call prepare(local_tokens) (or run it eagerly) first"
            )
        fn = self._launcher(cfg)
        m = m or next(m for m, c in self._cfgs.items() if c == cfg)
        args = self._launch_args(m, cfg, self.y, None, None, None)
        _preload_compiled(fn, *args, torch.cuda.current_stream())
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self._armed.add(cfg)

    def _launch_args(self, m, cfg, y, x, ids, tw, tail=None):
        sc = self._cfg_sched(cfg)
        peers = [int(b) for b in self.arena.base_ptrs] + [0] * (MAX_TP - self.tp)

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        return (
            self.w1.data_ptr(),
            self.w1s.data_ptr(),
            self.w2.data_ptr(),
            self.w2s.data_ptr(),
            ptr(x),
            ptr(ids),
            ptr(tw),
            y.data_ptr(),
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
            sc.xl_e0 if self._xl(cfg) else (0 if cfg.dyn else sc.piece_e0),
            0 if self.xg is None else self.xg.data_ptr(),
            0 if cfg.dyn else sc.col0,
            0 if cfg.dyn else sc.ncol,
            *(ptr(t) for t in (tail or (None, None, None))),
            self._qall.offset,
            self._sall.offset,
            self.n_cta,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        tail=None,
    ):
        """tail (ag_rs): (res_in, res_out, norm_w), bf16 [m, H], [m, H], [H]:
        also res_out = y + res_in and every rank's per-token FP8 rows of
        GemmaRMSNorm(res_out) (eps tn_eps) gathered: returns (y, q [M, H] fp8
        bytes, scale [M] fp32), views of arena buffers valid until the next
        forward with a tail."""
        m = self._local_tokens(x_local, topk_weights, topk_ids)
        rows = m * self.tp if self.ar else m  # output rows
        if out is not None and (
            tuple(out.shape) != (rows, self.H)
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(f"out: need a contiguous bf16 [{rows}, {self.H}]")
        if m == 0:
            return x_local.new_empty((0, self.H)) if out is None else out
        x = x_local.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        cfg = self.config(m)
        if tail is not None:
            if not self.tail_ok(m):
                raise ValueError("tail: ag_rs, the LB schedule, at most TN_MAX local tokens")
            res_in, res_out, nw = tail
            for t, shp in ((res_in, (m, self.H)), (res_out, (m, self.H)), (nw, (self.H,))):
                if tuple(t.shape) != shp or t.dtype != torch.bfloat16 or not t.is_contiguous():
                    raise ValueError(f"tail: need contiguous bf16 {shp}")
            cfg = dataclasses.replace(cfg, tn=self.tn_mode)
        # output written in place: ag_rs / rs, and ar when the all-reduce ends
        # locally (one-shot LL); else it lands in the peers' symmetric buffer
        local_y = not self.ar or self._arll(cfg) or self._ag8(cfg)
        y = out if out is not None and local_y else self.y
        tptr = tuple(t.data_ptr() for t in tail) if tail is not None else ()
        key = (m, x.data_ptr(), ids.data_ptr(), tw.data_ptr(), y.data_ptr(), cfg, tptr)
        if self._args[0] != key:
            args = self._launch_args(m, cfg, y, x, ids, tw, tail)
            # the inputs stay referenced while their pointers are cached
            self._args = (key, (args, x, ids, tw))
        self._arm(cfg, m)
        _run_compiled(
            self._launcher(cfg), *self._args[1][0], torch.cuda.current_stream()
        )
        if self.check and not torch.cuda.is_current_stream_capturing():
            self.check_errors()
        if not local_y:
            y = self._yall.local[:rows]
            return y if out is None else out.copy_(y)
        y = y[:rows] if out is None else out
        if tail is not None:
            T_ = m * self.tp
            return y, self._qall.local[: T_ * self.H].view(T_, self.H), self._sall.local[:T_]
        return y

    __call__ = forward

    def clear_errors(self) -> None:
        self.ctrl[CTRL_ERR] = 0

    def poll_errors(self) -> int:
        """OR of the watchdog codes of every wait that gave up (0 when healthy);
        a launch with a nonzero code produced invalid output. Synchronizes."""
        return int(self.ctrl[CTRL_ERR].item())

    def check_errors(self) -> None:
        """Raise (and clear) if a launch since the last check hit the watchdog.
        Synchronizes; runs after every forward with AITER_MEGAMOE_TP_CHECK=1."""
        err = self.poll_errors()
        if err:
            self.clear_errors()
            what = [n for b, n in _ERR_NAMES if err & b]
            raise RuntimeError(
                f"MegaMoeTP rank {self.rank}: a wait inside the kernel timed out "
                f"({'|'.join(what)}); the output is invalid. Ranks out of step "
                "(different call counts or local token counts) need reset()."
            )

    def reset(self) -> None:
        """Collective (every rank): drop all cross-rank state and restart the
        launch epoch, e.g. after a watchdog timeout or ranks calling forward a
        different number of times."""
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self.arena.storage.zero_()
        self.ctrl.zero_()
        self.routes.zero_()
        self.proutes.zero_()
        torch.cuda.synchronize(self.device)
        self.arena.barrier()

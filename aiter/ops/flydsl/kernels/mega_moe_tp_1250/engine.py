# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the gfx1250 MegaMoE TP kernel: symmetric arena, buffers, the dispatch
paths (small / mid / big M) and launches."""

from __future__ import annotations

import dataclasses
import os

import flydsl.expr as fx
import torch

from ..mega_moe_tp.mega_moe_tp import SymmetricArena
from ..tensor_shim import _run_compiled, ptr_arg
from .config import (
    CT_ERR,
    CTRL_INTS,
    FLAG_INTS,
    MAX_TP,
    TS_INTS,
    TS_N,
    GemmCfg,
    LaunchCfg,
    tcnt_ints,
)
from .kernel import compile_mega_moe_tp_1250

__all__ = ["MegaMoeTP1250Engine", "default_launch_cfg"]

# AR mode: above this many tokens Y is reduce-scatter + all-gather (MEGA_AR_TWO_MIN)
AR_TWO_MIN = int(os.environ.get("MEGA_AR_TWO_MIN", "512"))


def default_launch_cfg(tokens: int) -> LaunchCfg:
    """The launch config of a batch of (global) tokens: its dispatch path and tiles
    (MEGA_CFG="field=value,..." overrides LaunchCfg fields, for experiments)."""
    cfg = _default_launch_cfg(tokens)
    ov = os.environ.get("MEGA_CFG", "")
    for f in filter(None, ov.split(",")):
        k, v = f.split("=")
        if "." in k:  # g1.tile_m=128: a GemmCfg field
            g, k = k.split(".")
            sub = dataclasses.replace(getattr(cfg, g), **{k: int(v)})
            cfg = dataclasses.replace(cfg, **{g: sub})
        else:
            cfg = dataclasses.replace(cfg, **{k: type(getattr(cfg, k))(int(v))})
    return cfg


def _default_launch_cfg(tokens: int) -> LaunchCfg:
    if tokens <= 256:
        g = GemmCfg(tile_m=16, tile_n=256)
        # tuned (TP4, scripts/tune_mm.py): 2 CTAs per CU up to 64 tokens (cheaper
        # grid syncs, few tiles anyway), 3 from 128 (more weight streams in flight);
        # every CTA counts the routes itself (no barrier 1), static task order
        return LaunchCfg(
            "small", g, g, cta_per_cu=3, place_rpw=1, dyn=False, local_hist=True,
            df2=tokens <= 8,  # G2 -> RS by m-tile counts: only pays at the fewest tiles
            df0=tokens <= 64,  # placement -> G1 by expert counts (no barrier 2)
        )
    # mid / big: 128-row tiles re-read each expert's weights half as often; their
    # LDS (2 K-stages of 256) allows 2 CTAs per CU
    g = GemmCfg(tile_m=128, tile_n=256, tile_k=256, num_buffers=2)
    # from 4096 tokens: RS partials stay in the own arena and owners pull them
    # (scattered remote stores cost a fixed overhead per row), as int8 + e8m0 per
    # 32 (logits_diff ~7.5e-5 vs ~6.9e-6, gate 1e-2); below, both cost a little
    q8 = tokens >= 4096
    if tokens <= 2048:
        # 3 CTAs per CU (a 128-row tile's 424 VGPRs allow 2): smaller tiles, more
        # tiles in flight; 32-row tiles up to 1024 tokens, 64-row at 2048
        gs = GemmCfg(tile_m=32 if tokens <= 1024 else 64, tile_n=256, tile_k=256, num_buffers=2)
        return LaunchCfg("mid", gs, gs, cta_per_cu=3, place_rpw=1 if tokens <= 512 else 4)
    if tokens <= 8192:
        return LaunchCfg("mid", g, g, cta_per_cu=2, place_rpw=16, rs_pull=q8, rs_q8=q8)
    return LaunchCfg("big", g, g, cta_per_cu=2, place_rpw=32, rs_pull=True, rs_q8=q8)


class MegaMoeTP1250Engine:
    """ag_rs: forward(x [m, H] own rows, topk_weights / topk_ids [m, topk]) -> y [m, H],
    y = the TP layer's output rows of this rank's tokens."""

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
        activation: str = "swiglu",
        swiglu_limit: float | None = None,
        situ_beta: float = 1.0,
        situ_linear_beta: float = 1.0,
        group=None,
        device: torch.device | None = None,
        cfg_fn=default_launch_cfg,
    ):
        if not 1 <= world_size <= MAX_TP:
            raise ValueError(f"tp {world_size} not in [1, {MAX_TP}]")
        if model_dim % 1024 or inter_dim % 128:
            raise ValueError(f"h{model_dim} i{inter_dim}: need h % 1024, i % 128")
        self.rank, self.tp = int(rank), int(world_size)
        self.H, self.I, self.E, self.K = model_dim, inter_dim, experts, topk
        self.mmax = int(max_local_tokens)
        self.act = activation
        if swiglu_limit is None:
            swiglu_limit = 7.0 if activation == "swiglu" else float("inf")
        self.swiglu_limit = float(swiglu_limit)
        self.situ = (float(situ_beta), float(situ_linear_beta))
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(self.device)
        self.n_cu = int(props.multi_processor_count)
        self.cfg_fn = cfg_fn
        self.set_weights(w1, w1_scale, w2, w2_scale)

        tot, H, K = self.mmax * self.tp, model_dim, topk
        arena = SymmetricArena(group=group, device=self.device)
        self._xq = arena.reserve("xq", (tot, H // 2), torch.uint8)
        self._xs = arena.reserve("xs", (tot, H // 32), torch.uint8)
        self._ids = arena.reserve("ids", (tot, K), torch.int32)
        self._tw = arena.reserve("tw", (tot, K), torch.float32)
        self._recv = arena.reserve("recv", (self.tp, self.mmax, H), torch.bfloat16)
        # AR two-step Y: every rank's reduced rows, read by the others
        self._yred = arena.reserve("yred", (self.tp, self.mmax, H), torch.bfloat16)
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        arena.commit()
        self.arena = arena
        dev = self.device
        self.ctrl = torch.zeros(CTRL_INTS + TS_INTS, dtype=torch.int32, device=dev)
        # the routed-row buffers (contiguous-M, every expert's rows tile aligned)
        am = 256  # the largest tile_m of any LaunchCfg
        self.cm = (tot * K + experts * am + am - 1) // am * am
        self.a1 = torch.zeros((self.cm, H // 2), dtype=torch.uint8, device=dev)
        self.a1s = torch.zeros((self.cm, H // 32), dtype=torch.uint8, device=dev)
        self.a2 = torch.zeros((self.cm, inter_dim // 2), dtype=torch.uint8, device=dev)
        self.a2s = torch.zeros(
            (self.cm * (inter_dim // 32) + 4096,), dtype=torch.uint8, device=dev
        )
        self.gout = torch.empty((self.cm, H), dtype=torch.bfloat16, device=dev)
        self.psum = torch.zeros(1024, dtype=torch.int32, device=dev)
        self.rpos = torch.empty(tot * K, dtype=torch.int32, device=dev)
        # per-m-tile completion counts: [epoch parity][G1, G2][m-tile]
        self.tcnt = torch.zeros(tcnt_ints(self.cm), dtype=torch.int32, device=dev)
        self.y = torch.empty((self.mmax, H), dtype=torch.bfloat16, device=dev)
        self._launchers: dict = {}

    def set_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        self.w = tuple(t.contiguous() for t in (w1, w1_scale, w2, w2_scale))

    def launcher(self, cfg: LaunchCfg):
        fn = self._launchers.get(cfg)
        if fn is None:
            fn = compile_mega_moe_tp_1250(
                H=self.H,
                I=self.I,
                E=self.E,
                TOPK=self.K,
                tp=self.tp,
                mmax=self.mmax,
                act=self.act,
                n_cu=self.n_cu,
                cfg=cfg,
            )[0]
            self._launchers[cfg] = fn
        return fn

    def forward(self, x, topk_weights, topk_ids, out=None, cfg: LaunchCfg | None = None,
                _m_override=False):
        m = int(x.shape[0])
        if m > self.mmax and not _m_override:
            raise ValueError(f"{m} local tokens > max_local_tokens {self.mmax}")
        if out is None:
            out = self.y[:m]
        if m == 0:
            return out
        cfg = cfg or self.cfg_fn(m * self.tp)
        x = x.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        peers = [int(b) for b in self.arena.base_ptrs] + [0] * (MAX_TP - self.tp)
        w1, w1s, w2, w2s = self.w

        def p(t):
            return ptr_arg(t, fx.Int32)

        args = (
            p(w1),
            p(w1s),
            p(w2),
            p(w2s),
            p(self.a1),
            p(self.a1s),
            p(self.a2),
            p(self.a2s),
            p(self.gout),
            p(self.psum),
            self.psum.data_ptr(),
            self.a1.data_ptr(),
            self.a1s.data_ptr(),
            self.gout.data_ptr(),
            x.data_ptr(),
            ids.data_ptr(),
            tw.data_ptr(),
            out.data_ptr(),
            self.ctrl.data_ptr(),
            self.rpos.data_ptr(),
            self.tcnt.data_ptr(),
            *peers,
            self._xq.offset,
            self._xs.offset,
            self._ids.offset,
            self._tw.offset,
            self._recv.offset,
            self._flag.offset,
            self._yred.offset,
            self.rank,
            m,
            self.cm,
            self.swiglu_limit,
            self.situ[0],
            self.situ[1],
            torch.cuda.current_stream(),
        )
        _run_compiled(self.launcher(cfg), *args, specialization_key=cfg)
        return out

    __call__ = forward

    def phase_stamps(self) -> torch.Tensor:
        """Debug (MEGA_TS=1): [NCTA, TS_N] stamps of the last launch, us from each CTA's start."""
        ts = self.ctrl[CTRL_INTS:].view(torch.int64).view(-1, TS_N).cpu()
        ts = ts[: int((ts[:, 0] != 0).sum())]
        # each CTA from its own start: the steady counter is per XCD (offsets of ms)
        return (ts - ts[:, :1]).double() / 100.0

    def forward_ar(self, x, topk_weights, topk_ids, out=None, cfg: LaunchCfg | None = None):
        """AR mode: x [T, H], topk_* [T, topk] replicated on every rank -> y [T, H]
        replicated (the TP layer's output), as MoE + all-reduce. T <= tp * max_local_tokens."""
        T = int(x.shape[0])
        if T % self.tp:
            raise ValueError(f"{T} tokens: a multiple of tp={self.tp} expected")
        if T > self.mmax * self.tp:
            raise ValueError(f"{T} tokens > tp * max_local_tokens {self.mmax * self.tp}")
        if out is None:
            if getattr(self, "y_ar", None) is None:
                self.y_ar = torch.empty((self.mmax * self.tp, self.H), dtype=torch.bfloat16,
                                        device=self.device)
            out = self.y_ar[:T]
        cfg = cfg or self.cfg_fn(T)
        cfg = dataclasses.replace(cfg, ar=True, rs_pull=False, ag_split=False,
                                  ar_two=cfg.ar_two or T > AR_TWO_MIN)
        return self.forward(x, topk_weights, topk_ids, out=out, cfg=cfg, _m_override=True)

    def poll_errors(self) -> int:
        return int(self.ctrl[CT_ERR].item())

    def close(self) -> None:
        self.arena.close()

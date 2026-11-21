# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The megakernel's host side: MegaKernel (one compiled kernel bound to
a MegaMoE's arena and combine), SlotsMega, CompactMega and
CompactMegaSet."""

from __future__ import annotations

import os

from .config import (
    SLOTS_CTAS_PER_CU,
    STREAM_PRODUCERS,
    compact_caps,
    compact_ctl_region,
    compact_row_regions,
    compact_variants,
    slots_cap,
    slots_recv_cap,
    slots_tile,
    slots_tokens,
    stream_layout,
    stream_reason,
    stream_tile_cap,
    wire_row,
)
from .kernel import make_megakernel
from .protocol import (
    CTL_ERR,
    CTL_FAIL,
    RT_N,
    TR_N,
    ctl_words,
)


class MegaKernel:
    """A single-kernel MegaMoE bound to one MegaMoE's arena and combine:
    ``layout="slots"`` keeps fixed expert slots, ``layout="compact"`` the
    staged compact layout.
    Each has its own control block, so they share no protocol state."""

    def __init__(self, config, res, device, layout="slots", variant=None, ctl=None):
        """``variant``: compact's (recv_lo, recv_hi, tile, CTAs per CU), one
        of compact_variants(config); ``ctl``: its arena control region."""
        import torch

        from aiter.jit.utils.chip_info import get_cu_num

        self.config = config
        self.res = res
        self.layout = layout
        compact = layout == "compact"
        geo = config.geometry
        arena = res.arena
        epr = config.experts_per_rank
        self.row = wire_row(config)
        if compact:
            self.recv_lo, self.recv_hi, t, compact_ctas = variant
            self.tile = t
            rows = compact_caps(config, self.recv_hi, t["tile_m"])
            self.cap = rows
            self.mt_cap = rows // t["tile_m"]
            self.regions = (*compact_row_regions(), ctl)
            self.stream_why = stream_reason(config, t, compact_ctas)
            self.stream = self.stream_why is None
            if self.stream:
                lay = stream_layout(
                    npes=config.world_size,
                    experts_per_rank=epr,
                    max_tok_per_rank=geo.max_tokens_per_rank,
                    row=self.row,
                    tile_cap=stream_tile_cap(rows, t["tile_m"]),
                )
                self.route = torch.zeros(
                    lay["bytes"] // 4, dtype=torch.int32, device=device
                )
            else:
                self.route = torch.empty(
                    geo.max_tokens_per_rank * config.topk,
                    dtype=torch.int32,
                    device=device,
                )
        else:
            cap = self.cap = slots_cap(config)
            t = self.tile = slots_tile(slots_tokens(config))
            self.recv_cap = slots_recv_cap(config)
            rows = epr * cap
            self.mt_cap = epr * (cap // t["tile_m"])
            self.regions = ("slots_rows", "slots_rowmap", "slots_ctl")
            self.route = None
            self.stream = False
        self.rows = rows
        from aiter.ops.flydsl.grouped_moe_gfx1250 import get_wmma_m_rep

        rep2 = get_wmma_m_rep(t["tile_m"], t["tile_n2"], 1, 4, "gemm2")
        inter = geo.inter_dim
        # Zeroed once: the control region's counters, parity words and
        # generation flags. Done before construction returns: the peers' first
        # writes may land as soon as every rank is past its construction barrier.
        arena.zero(self.regions[2])
        torch.cuda.synchronize(device)
        self.a2 = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
        self.a2_scale = torch.empty(
            (rows // rep2, inter // 32 * rep2), dtype=torch.uint8, device=device
        )
        self.ctas_per_cu = compact_ctas if compact else SLOTS_CTAS_PER_CU
        self.workers = int(get_cu_num()) * self.ctas_per_cu
        # Per-call switch for in-process A/B against the staged path. All
        # ranks must flip it together (the paths share no protocol state).
        self.enabled = True
        self._ctl_words = ctl_words(
            experts_per_rank=epr,
            mt_cap=self.mt_cap,
            workers=self.workers,
            npes=config.world_size,
            compact=compact,
        )
        self.ctl = torch.zeros(
            self._ctl_words + 2 * TR_N * self.workers + 8 * RT_N,
            dtype=torch.int32,
            device=device,
        )
        self._traced = 0
        self._launch = None
        self._act_key = None
        self._lined_up = None

    def serves(self, recv_token_bound) -> bool:
        """Whether a step with this receive bound runs the megakernel: the
        same answer on every rank, since every rank passes the same bound."""
        if not self.enabled:
            return False
        bound = self._bound(recv_token_bound)
        if self.layout == "compact":
            return self.recv_lo < bound <= self.recv_hi
        return bound <= self.recv_cap

    def _bound(self, recv_token_bound) -> int:
        return (
            self.config.max_recv if recv_token_bound is None else int(recv_token_bound)
        )

    def _launcher(self, stage1_act):
        if self._launch is None or self._act_key != stage1_act:
            config, arena = self.config, self.res.arena
            geo = config.geometry
            self._launch = make_megakernel(
                rank=config.rank,
                npes=config.world_size,
                experts=geo.experts,
                experts_per_rank=config.experts_per_rank,
                topk=config.topk,
                model_dim=config.hidden_dim,
                inter_dim=geo.inter_dim,
                max_tok_per_rank=geo.max_tokens_per_rank,
                cap=self.cap,
                row_stride=self.row,
                off_disp=arena.offset(self.regions[0]),
                off_rowmap=arena.offset(self.regions[1]),
                off_ctl=arena.offset(self.regions[2]),
                off_comb_inp=arena.offset("comb_inp"),
                comb_slot_stride=config.combine_slot_stride_for(0),
                arena_handle=arena.handle,
                stage1_act=stage1_act,
                workers=self.workers,
                ctas_per_cu=self.ctas_per_cu,
                layout=self.layout,
                mt_cap=self.mt_cap,
                stream=self.stream,
                producers=STREAM_PRODUCERS,
                **self.tile,
            )
            self._act_key = stage1_act
        return self._launch

    def forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        w1,
        w2,
        w1_scale,
        w2_scale,
        recv_token_bound=None,
    ):
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import torch

        from aiter import ActivationType
        from aiter.ops.flydsl.grouped_moe_gfx1250 import _grouped_weight_uint8

        from ...tensor_shim import _preload_compiled, _run_compiled

        comp = self.config.recipe.compute
        swiglu = comp.activation == ActivationType.Swiglu
        stage1_act = 2 if swiglu else 1
        sl = (
            float(comp.swiglu_limit)
            if comp.swiglu_limit
            else (7.0 if swiglu else float("inf"))
        )
        tokens = int(hidden_states.shape[0])
        arena = self.res.arena
        out = self.res.combine_output

        def ptr(addr):
            # Every buffer here is at least 16 B aligned (torch, arena regions,
            # and the e8m0 plane one 3584 B payload into the wire row).
            return flyc.from_c_void_p(fx.Uint8, int(addr), assumed_align=16)

        tracing = (
            os.environ.get("AITER_DMK_TRACE", "0") != "0"
            and self._traced < 6
            and not torch.cuda.is_current_stream_capturing()
        )
        if tracing:
            self.ctl[self._ctl_words :].zero_()
        disp = arena.local_ptr(self.regions[0])
        if self.layout == "compact":
            # The GEMMs see the rows this step's bound can fill; senders and
            # receiver drop anything past it alike.
            rows = min(
                compact_caps(
                    self.config, self._bound(recv_token_bound), self.tile["tile_m"]
                ),
                self.rows,
            )
            route = self.route.data_ptr()
        else:
            rows, route = self.rows, 0
        w1_u8 = _grouped_weight_uint8(w1)
        w2_u8 = _grouped_weight_uint8(w2)
        launch = self._launcher(stage1_act)
        args = (
            ptr(self.a2.data_ptr()),
            ptr(disp),
            ptr(disp + self.config.hidden_dim // 2),
            ptr(w1_u8.data_ptr()),
            ptr(w1_scale.data_ptr()),
            ptr(self.a2_scale.data_ptr()),
            ptr(w2_u8.data_ptr()),
            ptr(w2_scale.data_ptr()),
            ptr(arena.local_ptr(self.regions[1])),
            arena.handle,
            hidden_states.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            self.ctl.data_ptr(),
            tokens,
            rows,
            route,
            sl,
            fx.Stream(torch.cuda.current_stream()),
        )
        if launch is not self._lined_up:
            # Every rank compiles its own kernel (the rank is baked in), seconds
            # apart: line the ranks up before the first launch, so its waits do
            # not start long before a peer's kernel does.
            _preload_compiled(launch, *args)
            arena._communicator.barrier()
            self._lined_up = launch
        _run_compiled(launch, *args)
        if tracing:
            self._traced += 1
            self._print_trace(tokens)
        return self.res.output(tokens)

    def report(self):
        """AITER_DMK_TRACE: the last launch's phase stamps (graph replays too)."""
        if os.environ.get("AITER_DMK_TRACE", "0") != "0":
            self._print_trace(-1)

    def _print_trace(self, tokens):
        import torch

        torch.cuda.synchronize()
        tr = self.ctl[self._ctl_words : self._ctl_words + 2 * TR_N * self.workers]
        tr = tr.view(torch.int64).view(self.workers, TR_N)
        dump = os.environ.get("AITER_DMK_TRACE_DUMP", "")
        if dump:
            torch.save(
                tr.cpu(), f"{dump}_r{self.config.rank}_{self.layout}_{self._traced}.pt"
            )
        tr = tr.cpu().double()
        t0 = tr[:, 0].min()
        rel = (tr - t0) / 100.0  # steady counter: 100 MHz -> us
        names = [
            "start",
            "ticket",
            "produced",
            "planned",
            "gemm1",
            "gemm2",
            "flags",
            "end",
            "p_ready",
            "gate_seen",
            "p_issued",
            "p_retired",
            "g2_wait0",
            "g2_wait1",
        ]
        lead = ["src_seen", "plan_lds", "reps_done", "gates_p", "cf_seen", "gates_c"]
        cols = []
        for k, n in enumerate(names):
            c = rel[:, k]
            # Stamps only some CTAs take may be stale from an earlier launch.
            c = c[tr[:, k] >= t0] if k >= 8 else c
            if c.numel() == 0:
                continue
            cols.append(f"{n}={c.min():.1f}/{c.median():.1f}/{c.max():.1f}")
        rt = self.ctl[self._ctl_words + 2 * TR_N * self.workers :].view(torch.int64)
        rt = rt.view(RT_N, 4).cpu().double()
        ok = (rt[:, 0] >= t0) & (rt[:, 1] >= rt[:, 0])
        if ok.any():
            r_ = rt[ok]
            lat = (r_[:, 1] - r_[:, 0]) / 100.0
            iss = (r_[:, 0] - t0) / 100.0
            for d in sorted(set(r_[:, 3].long().tolist())):
                m = r_[:, 3] == d
                cols.append(
                    f"atom->r{d}:n={int(m.sum())},issue={iss[m].min():.1f}-{iss[m].max():.1f},"
                    f"lat={lat[m].min():.1f}/{lat[m].median():.1f}/{lat[m].max():.1f}"
                )
        cols.append(
            "cta0:" + ",".join(f"{n}={rel[0, 14 + k]:.1f}" for k, n in enumerate(lead))
        )
        if self.layout == "compact":
            x1 = rel[:, 23][tr[:, 23] >= t0]
            cols.append(
                f"x1:tally={x1.max() if x1.numel() else -1:.1f},"
                + ",".join(
                    f"{n}={rel[0, 20 + k]:.1f}"
                    for k, n in enumerate(("x1_seen", "srep", "gates_s"))
                )
            )
        # The CTAs that finish GEMM2 last: their last unit's wait.
        late = rel[:, 5].argsort(descending=True)[:3].tolist()
        cols.append(
            "late="
            + ",".join(
                f"b{b}:g1={rel[b, 4]:.0f},w0={rel[b, 12]:.0f},w1={rel[b, 13]:.0f},g2={rel[b, 5]:.0f}"
                for b in late
            )
        )
        print(
            f"[dmk trace r{self.config.rank} T={tokens}] (min/med/max us) "
            + " ".join(cols),
            flush=True,
        )

    def errors(self) -> int:
        """Routes dropped for a full expert slot since construction."""
        return int(self.ctl[CTL_ERR].item())

    def fails(self) -> list:
        """AITER_DMK_NOTRAP: timed-out waits per site (X1, send gate, source
        flags, plan gate, g1_ready, combine flags, reduce gate, rows gate)."""
        return self.ctl[CTL_FAIL : CTL_FAIL + 8].tolist()


class SlotsMega(MegaKernel):
    """The slots megakernel: fixed per-expert slots, for small steps."""

    def __init__(self, config, res, device):
        super().__init__(config, res, device, layout="slots")


class CompactMega(MegaKernel):
    """One compact megakernel: ``variant`` is a row of compact_variants(config),
    ``ctl`` its arena control region."""

    def __init__(self, config, res, device, variant, ctl):
        super().__init__(
            config, res, device, layout="compact", variant=variant, ctl=ctl
        )


class CompactMegaSet:
    """One CompactMega per row of COMPACT_TILES, each over its own
    receive bounds (compact_variants) with its own control region; a step runs
    the one its bound falls in. They share the staged path's rows, rowmap and
    comb_inp the way compact and the staged path do: a rank writes a peer's
    copy only after a message from that peer's current launch.
    Attributes not defined here are those of the kernel for the largest steps."""

    def __init__(self, config, res, device):
        self.variants = [
            CompactMega(config, res, device, v, compact_ctl_region(i))
            for i, v in enumerate(compact_variants(config))
        ]

    @property
    def enabled(self) -> bool:
        return all(v.enabled for v in self.variants)

    @enabled.setter
    def enabled(self, on: bool) -> None:
        for v in self.variants:
            v.enabled = on

    def for_bound(self, recv_token_bound):
        """The kernel that runs a step with this receive bound, or None."""
        for v in self.variants:
            if v.serves(recv_token_bound):
                return v
        return None

    def serves(self, recv_token_bound) -> bool:
        return self.for_bound(recv_token_bound) is not None

    def forward(self, hidden_states, topk_weights, topk_ids, **kw):
        mk = self.for_bound(kw.get("recv_token_bound"))
        return mk.forward(hidden_states, topk_weights, topk_ids, **kw)

    def errors(self) -> int:
        return sum(v.errors() for v in self.variants)

    def fails(self) -> list:
        return [sum(site) for site in zip(*(v.fails() for v in self.variants))]

    def report(self):
        for v in self.variants:
            v.report()

    def __getattr__(self, name):
        if name == "variants":
            raise AttributeError(name)
        return getattr(self.variants[-1], name)

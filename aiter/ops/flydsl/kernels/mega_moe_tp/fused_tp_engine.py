# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel TP MegaMoE (:mod:`.fused_tp`).

One launch per forward: quantize + AllGather, route scan, GEMM1, activation,
MXFP4 requant, GEMM2, top-k reduce and ReduceScatter all run inside one FlyDSL
kernel, and the GEMM1 -> GEMM2 intermediate stays in LDS.
"""

from __future__ import annotations

import os

import torch

from ..tensor_shim import _run_compiled
from .fused_tp import (
    CTRL_INTS,
    FLAG_INTS,
    MAX_TP,
    NCTA_MAX,
    compile_fused_tp,
    fused_tp_supported,
)
from .symmetric_arena import SymmetricArena

__all__ = ["FusedTpMegaMoe", "fused_tp_supported"]


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
        group=None,
        device: torch.device | None = None,
    ):
        if not fused_tp_supported(model_dim, inter_dim, world_size):
            raise ValueError(
                f"fused TP MegaMoE does not tile h{model_dim} i{inter_dim} tp{world_size}"
            )
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
        self._ids = arena.reserve("ids", (tot, K), torch.int32)
        self._w = arena.reserve("w", (tot, K), torch.float32)
        # ReduceScatter receive slots [source rank][row][H]. A peer only writes
        # launch n + 1's rows after this rank joined n + 1's AllGather.
        self._recv = arena.reserve("recv", (self.tp, self.mmax, H), torch.bfloat16)
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        arena.commit()
        self.arena = arena
        self.ctrl = torch.zeros(CTRL_INTS + 16 + 256 * 1024, dtype=torch.int32, device=self.device)
        # Per-route GEMM2 rows (weighted, bf16); the comm waves reduce them.
        self.routes = torch.empty(
            (tot * topk + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.y = torch.empty((self.mmax, H), dtype=torch.bfloat16, device=self.device)

        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        # every CTA sends ceil(mmax / n_cta) of this rank's tokens in the AllGather
        self.agr = max(1, -(-self.mmax // self.n_cta))
        if self.n_cta > NCTA_MAX:
            raise ValueError(f"at most {NCTA_MAX} CTAs (AllGather flag layout)")
        self._units, self._cta_units = self._schedule()
        # K-slice partials of split experts' pieces 1..P-1, one route-row set each
        # (piece 0 writes the route's own row); the push adds them up.
        extra = max(self._npieces - 1, 0) * tot * topk
        if extra * H * 2 >= 1 << 31:
            raise ValueError("split-expert partial rows exceed 32-bit buffer offsets")
        self.proutes = torch.empty(
            (extra + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self._args: dict = {}
        # E4M3 route rows (the split path's FP8 stage-2 route-out numerics):
        # half the GEMM2 -> ReduceScatter traffic, split-level accuracy.
        # AITER_MEGAMOE_ROUTE_FP8=0 keeps bf16 routes (~10x lower error).
        self.route_fp8 = os.environ.get("AITER_MEGAMOE_ROUTE_FP8", "1") == "1"
        # E4M3 ReduceScatter wire format: half the xGMI bytes, error vs torch
        # ~0.038 (split path: 0.034-0.044); AITER_MEGAMOE_RS_FP8=0: bf16.
        self.rs_fp8 = os.environ.get("AITER_MEGAMOE_RS_FP8", "1") == "1"

    # -- work schedule -----------------------------------------------------
    def _pieces(self, rem: int) -> int:
        nb = self.I // 128
        options = [1] + ([2] if nb % 2 == 0 and nb > 2 else []) + [nb]
        for p in options:
            if rem * p >= self.n_cta or p == nb:
                return p
        return nb

    def _schedule(self):
        """Every CTA takes E // n_cta whole experts; the leftover experts are cut
        into inter slices dealt one per CTA (their GEMM2 K-slice partials are
        summed by the ReduceScatter push)."""
        E, C, I = self.E, self.n_cta, self.I
        full, rem = divmod(E, C)
        per = [[] for _ in range(C)]
        e = 0
        for c in range(C):
            for _ in range(full):
                per[c].append((e, 0, I))
                e += 1
        if rem:
            p = self._pieces(rem)
            width = I // p
            pieces = [(e + j, k * width, width) for j in range(rem) for k in range(p)]
            for idx, piece in enumerate(pieces):
                per[(C - 1 - idx) % C].append(piece)
        self._npieces = p if rem else 0
        self._piece_e0 = full * C
        units, starts = [], [0]
        for lst in per:
            for e0, i0, icnt in lst:
                units.append([e0, i0, icnt, 0])
            starts.append(len(units))
        dev = self.device
        return (
            torch.tensor(units, dtype=torch.int32, device=dev),
            torch.tensor(starts, dtype=torch.int32, device=dev),
        )

    def _mt(self, local_tokens: int) -> int:
        routes = local_tokens * self.tp * self.K
        rpe = (routes + self.E - 1) // self.E
        return max(1, min(6, (rpe + 15) // 16))

    def _launcher(self, mt: int):
        return compile_fused_tp(
            H=self.H,
            I=self.I,
            TOPK=self.K,
            MT=mt,
            TMAX=self.mmax * self.tp,
            act=self.activation,
            situ_beta=self.situ[0],
            situ_linear_beta=self.situ[1],
            npieces=self._npieces,
            route_fp8=self.route_fp8,
            rs_fp8=self.rs_fp8,
            agr=self.agr,
            tp=self.tp,
        )

    def forward(
        self, x_local: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ):
        m = int(x_local.shape[0])
        if m > self.mmax:
            raise ValueError(f"{m} local tokens exceed max_local_tokens {self.mmax}")
        x = x_local.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        key = (m, x.data_ptr(), ids.data_ptr(), tw.data_ptr())
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
                self._units.data_ptr(),
                self._cta_units.data_ptr(),
                *peers,
                self._x.offset,
                self._xs.offset,
                self._ids.offset,
                self._w.offset,
                self._recv.offset,
                self._flag.offset,
                self.rank,
                self.tp,
                m,
                self.mmax,
                int(self._npieces),
                int(self._piece_e0),
                self.n_cta,
            )
            self._args[key] = (args, (x, ids, tw))
        else:
            args = args[0]
        _run_compiled(self._launcher(self._mt(m)), *args, torch.cuda.current_stream())
        return self.y[:m]

    __call__ = forward

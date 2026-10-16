# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel EP MegaMoE: launch configs, launches, and the
pure-torch reference oracle used to validate the kernel (对拍)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.utility import fp4_utils

from ..mega_moe.quant import per_1x32_mx_quant
from ..mega_moe_tp.mega_moe_tp import SymmetricArena
from ..tensor_shim import _run_compiled
from .mega_moe_ep_config import (
    CTRL_INTS,
    FLAG_INTS,
    MAX_NPES,
    ep_shape_supported,
)
from .mega_moe_ep_kernel import compile_mega_moe_ep

__all__ = ["LaunchCfg", "MegaMoeEPEngine", "ep_reference"]

LDS_LIMIT = 160 * 1024
# Source-indexed payload at and above this MTPR class (mirrors the dispatch fix in
# mega_moe/mega_moe_config.INDEXED_PAYLOAD_MIN_MTPR): the receive row = rank*mtpr +
# token is bounded by npes*mtpr, overflow-proof under any cross-rank routing skew.
INDEXED_PAYLOAD_MIN_MTPR = 32768


def _dequant_expert(weight, scale, rows, cols):
    """Dequantize one MXFP4 expert weight (packed FP4 values * per-1x32 E8M0 scale)."""
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather_variable(tensor, token_counts=None):
    """All-gather rows across ranks with per-rank variable counts (pad, gather, trim)."""
    world = dist.get_world_size()
    if token_counts is None:
        local = torch.tensor(tensor.shape[0], dtype=torch.int64, device=tensor.device)
        counts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(counts, local)
        token_counts = [int(c.item()) for c in counts]
    max_count = max(token_counts)
    if tensor.shape[0] < max_count:
        padding = torch.empty(
            (max_count - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding))
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor)
    return torch.cat([v[:c] for v, c in zip(gathered, token_counts)]), token_counts


@torch.no_grad()
def ep_reference(
    x,
    route_weights,
    ids,
    ref_weights,
    *,
    rank,
    world_size,
    model_dim,
    inter_dim,
    experts,
    swiglu_limit,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    """Pure-torch EP MoE oracle matching MegaMoEV2 numerics: dispatch (all-gather +
    route to the expert owner) -> per-local-expert GEMM1 + act + GEMM2 scaled by the
    top-k weight -> combine (all-reduce the partials, keep this rank's rows).

    ``ref_weights`` are the *unshuffled* quantized weights
    ``(w1_q, w1_scale, w2_q, w2_scale)`` as handed to MegaMoEV2 (see the op-test's
    ``_quantize_weights``). Collective when torch.distributed is initialized and
    ``world_size > 1``; otherwise a single-process local MoE (all experts local).
    """
    distributed = world_size > 1 and dist.is_available() and dist.is_initialized()
    if distributed:
        x_all, token_counts = _all_gather_variable(x)
        weights_all, _ = _all_gather_variable(route_weights, token_counts)
        ids_all, _ = _all_gather_variable(ids, token_counts)
    else:
        x_all, weights_all, ids_all = x, route_weights, ids
        token_counts = [x.shape[0]]
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = experts // world_size
    expert_start = rank * local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_start + local_experts)]
    for expert in active.tolist():
        positions = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = positions[:, 0], positions[:, 1]
        lid = expert - expert_start
        w1 = _dequant_expert(w1_q[lid], w1_scale[lid], 2 * inter_dim, model_dim)
        w2 = _dequant_expert(w2_q[lid], w2_scale[lid], model_dim, inter_dim)
        inp = x_all[rows].float()
        gate = inp @ w1[:inter_dim].T
        up = inp @ w1[inter_dim:].T
        if act == "situv2":
            gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
            hidden = gate * up
        else:
            if swiglu_limit > 0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            hidden = F.silu(gate) * up
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
    if distributed:
        dist.all_reduce(partial)
        start = sum(token_counts[:rank])
        return partial[start : start + x.shape[0]]
    return partial


@dataclass(frozen=True)
class LaunchCfg:
    """One EP kernel variant: row tile 16*mt; ``indexed_payload`` source-indexed
    receive layout (MTPR class >= INDEXED_PAYLOAD_MIN_MTPR); ``a8`` MXFP8 activations."""

    mt: int
    indexed_payload: bool = False
    a8: bool = False


class MegaMoeEPEngine:
    """Host side of the fused EP MoE (see the module docstring).

    SCAFFOLD STATE: ``forward`` is not wired to the device kernel yet (the role bodies
    are stubs -- see the planner / arena_flags / dispatch_send / compute / combine
    milestones). Use ``reference_forward`` / ``ep_reference`` as the numerical oracle;
    ``config`` / ``_launcher`` already select and compile the one-launch, grid=#CU,
    block=512 launcher so the host plumbing can be exercised.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        max_tok_per_rank,
        w1,
        w1_scale,
        w2,
        w2_scale,
        quant="a4w4",
        act="silu",
        situ_beta=1.0,
        situ_linear_beta=1.0,
        swiglu_limit=0.0,
        group=None,
        device=None,
    ):
        if quant not in ("a4w4", "a8w4"):
            raise ValueError("MegaMoeEP quant must be 'a4w4' or 'a8w4'")
        if not ep_shape_supported(model_dim, inter_dim, experts, world_size):
            raise ValueError(
                f"fused EP MegaMoE does not tile h{model_dim} i{inter_dim} "
                f"e{experts} np{world_size}"
            )
        mtpr = int(max_tok_per_rank)
        if mtpr <= 0 or mtpr & (mtpr - 1):
            raise ValueError(f"max_tok_per_rank={mtpr} must be a power of two")
        if not 0 < topk <= 16:
            raise ValueError(f"MegaMoeEP topk must be in [1, 16], got {topk}")
        if act not in ("silu", "situv2"):
            raise ValueError(f"MegaMoeEP act must be 'silu' or 'situv2', got {act!r}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.H = int(model_dim)
        self.I = int(inter_dim)
        self.E = int(experts)
        self.epr = int(experts // world_size)
        self.K = int(topk)
        self.mtpr = mtpr
        self.a8 = quant == "a8w4"
        self.act = act
        self.situ = (float(situ_beta), float(situ_linear_beta))
        self.swiglu_limit = float(swiglu_limit)
        self.group = group
        if device is None:
            device = torch.device("cuda", self.rank)
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, self.rank)
        self.device = device
        self._bind_weights(w1, w1_scale, w2, w2_scale)
        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        if self.gfx != "gfx950":
            raise ValueError(f"fused EP MegaMoE needs gfx950, not {self.gfx}")
        self._cfgs: dict = {}
        self._launchers: dict = {}
        # hipIpc symmetric arena: dispatch deposits each routed token's payload here on
        # the expert-owner rank (indexed layout, world_size*mtpr rows); combine deposits
        # the weighted GEMM2 partials back on the source rank; the epoch flags live in it
        # too. Reserved in the same order on every rank (collective commit maps peers).
        tot = self.world_size * self.mtpr
        a_row = self.H if self.a8 else self.H // 2
        arena = SymmetricArena(group=group, device=self.device)
        self._payload = arena.reserve("payload", (tot, a_row), torch.uint8)
        self._pscale = arena.reserve("pscale", (tot, self.H // 32), torch.uint8)
        self._recv = arena.reserve("recv_le", (tot * self.K,), torch.int32)
        self._part = arena.reserve("part", (self.mtpr * self.K, self.H), torch.bfloat16)
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        self._out = arena.reserve("out", (self.mtpr, self.H), torch.bfloat16)
        arena.commit()
        self.arena = arena
        self.ctrl = torch.zeros(CTRL_INTS, dtype=torch.int32, device=self.device)
        # Epoch starts at 1: the arena ready-flags are zero-initialized and the waits are
        # serial-number ``flag >= epoch`` (wrap-safe ``before``), so epoch 0 would make
        # every dispatch_wait / combine_wait pass instantly (0 >= 0) and all cross-wave /
        # cross-rank ordering would collapse into a race.
        self.epoch = 1
        # Per-CTA GEMM scratch (local, NOT peer-visible): GEMM1 writes the E8M0-quantized
        # intermediate (inter / inters), GEMM2 reads it + writes model-dim bf16 (gout) for
        # combine_send. Indexed bid*RG*row in the kernel; sized for the largest row tile
        # (mt=3 -> RG=48) so a smaller-RG launch indexes a strict prefix.
        rg_max = 48
        a_pack = 1 if self.a8 else 2
        self._inter = torch.empty(
            self.n_cta * rg_max * (self.I // a_pack),
            dtype=torch.uint8,
            device=self.device,
        )
        self._inters = torch.empty(
            self.n_cta * rg_max * (self.I // 32), dtype=torch.uint8, device=self.device
        )
        self._gout = torch.empty(
            self.n_cta * rg_max * self.H, dtype=torch.bfloat16, device=self.device
        )
        # Per-CTA GEMM1 bridge: the planner's gathered payload rows (L_RIX[r]//topk)
        # written to global so the V2 ATileLoader / AScaleLoader source-key gather can
        # read them (indexed_input row_map). Indexed bid*RG+r in the kernel.
        self._rowmap = torch.empty(
            self.n_cta * rg_max, dtype=torch.int32, device=self.device
        )

    def _bind_weights(self, w1, w1_scale, w2, w2_scale):
        # GEMM1 is the V2 pipelined path: its BWeightLoader / BScaleLoader read W1 and
        # its E8M0 scale in the shuffle_weight_a16w4 / shuffle_scale_a16w4 (gate|up
        # interleave) B-layout, so shuffle them host-side (natural weights are fed by
        # the tests / oracle). GEMM2 stays the naive natural-layout read -> W2 natural.
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        w1 = w1 if w1.is_contiguous() else w1.contiguous()
        self.w1 = shuffle_weight_a16w4(w1, 16, True).contiguous()
        self.w1s = shuffle_scale_a16w4(
            w1_scale.reshape(-1, self.H // 32).contiguous(), self.epr, True
        ).contiguous()
        self.w2, self.w2s = (
            t if t.is_contiguous() else t.contiguous() for t in (w2, w2_scale)
        )

    def set_weights(self, w1, w1_scale, w2, w2_scale):
        self._bind_weights(w1, w1_scale, w2, w2_scale)

    def _fit_mt(self, m):
        # The V2 pipelined GEMM1 packs rows in pairs (MfmaScaleGU / AScaleLoader use
        # m_repeat//2 scale groups), so the row tile mt (=m_repeat) must be EVEN. mt=2
        # (RG=32) fits the rg_max=48 per-CTA scratch and all shapes' LDS; larger even
        # tiles (mt=4) are a perf follow-up (they need rg_max>=64).
        del m
        return 2

    def config(self, m):
        cfg = self._cfgs.get(m)
        if cfg is None:
            cfg = LaunchCfg(
                mt=self._fit_mt(m),
                indexed_payload=self.mtpr >= INDEXED_PAYLOAD_MIN_MTPR,
                a8=self.a8,
            )
            self._cfgs[m] = cfg
        return cfg

    def _launcher(self, cfg):
        fn = self._launchers.get(cfg)
        if fn is None:
            fn = compile_mega_moe_ep(
                H=self.H,
                I=self.I,
                TOPK=self.K,
                MT=cfg.mt,
                TMAX=self.mtpr * self.world_size,
                E=self.E,
                EPR=self.epr,
                NPES=self.world_size,
                act=self.act,
                situ_beta=self.situ[0],
                situ_linear_beta=self.situ[1],
                swiglu_limit=self.swiglu_limit,
                a8=cfg.a8,
                indexed_payload=cfg.indexed_payload,
            )
            self._launchers[cfg] = fn
        return fn

    def reference_forward(self, x_bf16, wts, topk_ids, ref_weights):
        """Run the pure-torch oracle for these inputs (对拍). ``ref_weights`` are the
        unshuffled quantized weights ``(w1_q, w1_scale, w2_q, w2_scale)``."""
        return ep_reference(
            x_bf16,
            wts,
            topk_ids,
            ref_weights,
            rank=self.rank,
            world_size=self.world_size,
            model_dim=self.H,
            inter_dim=self.I,
            experts=self.E,
            swiglu_limit=self.swiglu_limit,
            act=self.act,
            situ_beta=self.situ[0],
            situ_linear_beta=self.situ[1],
        )

    def _launch_args(self, m, x, xs, ids, wts, out):
        """The kernel's positional args (minus stream): weights, local input, output,
        global ctrl, arena peer bases, arena region offsets, scalars, grid (#CU)."""
        peers = [int(b) for b in self.arena.base_ptrs]
        peers += [0] * (MAX_NPES - self.world_size)

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        return (
            self.w1.data_ptr(),
            self.w1s.data_ptr(),
            self.w2.data_ptr(),
            self.w2s.data_ptr(),
            ptr(x),
            ptr(xs),
            ptr(ids),
            ptr(wts),
            out.data_ptr(),
            self.ctrl.data_ptr(),
            *peers,
            self._payload.offset,
            self._pscale.offset,
            self._recv.offset,
            self._part.offset,
            self._flag.offset,
            self._out.offset,
            self._inter.data_ptr(),
            self._inters.data_ptr(),
            self._gout.data_ptr(),
            self._rowmap.data_ptr(),
            self.rank,
            self.world_size,
            self.epr,
            m,
            self.mtpr,
            self.epoch,
            self.n_cta,
        )

    def forward(
        self,
        x_bf16,
        wts,
        topk_ids,
        *,
        stream=None,
        slice_output=True,
        mask_invalid_slots=None,
    ):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        del mask_invalid_slots  # TODO(C): expected-route masking in combine_final
        fn = self._launcher(self.config(run_tokens))
        # Dynamic MX quant of the activations (same per-1x32 MX path as MegaMoEV2); the
        # scales are natural [m, H/32], which dispatch copies verbatim into pscale.
        x_q, xs = per_1x32_mx_quant(
            x_bf16.contiguous(), quant_mode="fp8" if self.a8 else "fp4"
        )
        out = torch.zeros((self.mtpr, self.H), dtype=torch.bfloat16, device=self.device)
        if stream is None:
            stream = torch.cuda.current_stream()
        # recv_le init (SENTINEL = epr, masked by the planner) on the kernel stream, so it
        # is ordered before the launch; the kernel's barrier_init orders it cross-rank.
        # No host sync / dist.barrier -> forward() is a pure stream pipeline.
        with torch.cuda.stream(stream):
            self._recv.local.fill_(self.epr)
        _run_compiled(
            fn, *self._launch_args(run_tokens, x_q, xs, topk_ids, wts, out), stream
        )
        self.epoch += 1
        return out[:run_tokens] if slice_output else out

    __call__ = forward

    # --- watchdog / lifecycle (parity with MegaMoeTP; fully wired with the arena) ----
    def poll_errors(self):
        return int(self.ctrl[0].item())

    def error_flag(self):
        return self.ctrl[0:1]

    def clear_errors(self):
        self.ctrl[0] = 0

    def reset(self):
        self.ctrl.zero_()
